#!/usr/bin/env python3
# pyre-ignore-all-errors
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: B023, E501, F841

import asyncio
import os
import sys
from typing import Any, cast
from uuid import uuid4

from genkit import ActionRunContext, Genkit, Message
from genkit.agent import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentResult,
    InMemoryLatestStateStore,
    SessionRunner,
    TurnResult,
)
from genkit.plugins.google_genai import GoogleAI

# Initialize Genkit. GoogleAI plugin automatically utilizes the GEMINI_API_KEY environment variable.
ai = Genkit(plugins=[GoogleAI()])

# 1. Deterministic Fast-Path Keywords (0ms LLM Overhead, 100% Reliable)
FAST_PATH_KEYWORDS = {'stop', 'cancel', 'halt', 'abort', 'exit'}


# Helper to mock plan parsing
def parse_plan(text: str) -> list[dict]:
    # Extremely simple parser for the sake of the PoC
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    steps = []
    for i, line in enumerate(lines):
        if line.startswith(('- ', '* ', '1.', '2.', '3.', '4.', '5.')):
            instr = line.lstrip('-*0123456789. ')
            steps.append({'instruction': instr, 'status': 'pending', 'file': f'src_file_{i + 1}.py'})
    if not steps:
        # Fallback if the LLM output format varies
        steps = [{'instruction': text, 'status': 'pending', 'file': 'workspace_code.py'}]
    return steps


async def execute_tool_step(step: dict, rollback_state: dict):
    """Simulates a worker executing a physical task (e.g. disk write, compilation)."""
    instruction = step['instruction']
    target_file = step.get('file', 'workspace_code.py')
    print(f'\n[Worker] Executing: {instruction}')

    # Track the file we are about to modify for the rollback handler
    rollback_state['written_files'].append(target_file)

    # Simulate a slow progress bar to allow real-time interruption
    for i in range(1, 6):
        await asyncio.sleep(0.6)  # Total 3 seconds
        print(f'  [Worker] [{target_file}] Progress: {"█" * i}{"░" * (5 - i)} ({i * 20}%)')

    print(f'[Worker] Success: Saved changes to {target_file}')
    return f'Wrote code for: {instruction}'


async def rollback_workspace(rollback_state: dict):
    """Simulates reverting the workspace to the last good checkpoint on cancellation."""
    print('\n[Rollback] !!! CRITICAL INTERRUPTION !!!')
    if not rollback_state['written_files']:
        print('[Rollback] No pending side-effects to revert.')
        return

    print(f'[Rollback] Reverting changes to: {rollback_state["written_files"]}')
    for file in rollback_state['written_files']:
        await asyncio.sleep(0.4)
        print(f'  [Rollback] Rolled back {file} to last committed state.')
    rollback_state['written_files'].clear()
    print('[Rollback] Workspace successfully restored to safety.')


# Stateful agent runner
store = InMemoryLatestStateStore()


async def supervised_agent_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        # State schema
        state_dict: dict[str, Any] = await sess.get_custom() or {
            'plan': [],
            'current_step_index': 0,
            'initialized': False,
        }
        plan: list[Any] = cast(list[Any], state_dict['plan'])
        current_step_index: int = cast(int, state_dict['current_step_index'])
        initialized: bool = cast(bool, state_dict['initialized'])

        async def save_state() -> None:
            state_dict['plan'] = plan
            state_dict['current_step_index'] = current_step_index
            state_dict['initialized'] = initialized
            await sess.update_custom(lambda _: state_dict)

        # Local mock rollback state for this session run
        rollback_state = {'written_files': []}

        # 1. INITIALIZATION: Build the initial plan if starting fresh
        if not initialized:
            print('\n[Supervisor] Analyzing initial goal and generating plan...')
            user_text = Message(inp.message).text if inp.message else ''
            plan_resp = await ai.generate(
                model='googleai/gemini-flash-latest',
                system="Create a short, logical 3-step plan to achieve the user's goal. Keep instructions concise.",
                prompt=user_text,
            )
            plan = parse_plan(plan_resp.text)
            current_step_index = 0
            initialized = True
            await save_state()
            print(
                '[Supervisor] Generated Plan:\n'
                + '\n'.join([f'  {i + 1}. {s["instruction"]} (Target: {s["file"]})' for i, s in enumerate(plan)])
            )

        # 2. THE MAIN ORCHESTRATION LOOP
        while current_step_index < len(plan):
            current_step = plan[current_step_index]

            # Wrap the step execution in a task so it can be cancelled
            worker_task = asyncio.create_task(execute_tool_step(current_step, rollback_state))
            feedback_event = asyncio.Event()

            # The Asynchronous Supervisor Monitor Task
            async def monitor_queue() -> tuple[str, str] | tuple[str, None]:
                while True:
                    if sess._intake.empty():
                        await asyncio.sleep(0.1)
                        continue

                    queued_input = await sess._intake.get()
                    if not (isinstance(queued_input, AgentInput) and queued_input.message):
                        continue

                    # Signal that we have intercepted feedback and are evaluating
                    feedback_event.set()

                    user_feedback = Message(queued_input.message).text.strip()
                    print(f"\n[Supervisor] Intercepted incoming feedback: '{user_feedback}'")

                    # --- HYBRID ROUTING ---
                    # A. FAST PATH (Deterministic): Instant Abort (0ms latency, 100% reliable)
                    if any(kw in user_feedback.lower() for kw in FAST_PATH_KEYWORDS):
                        print('[Supervisor] [FAST PATH] Hard kill keyword detected. Aborting immediately.')
                        worker_task.cancel()
                        await sess.add_messages(queued_input.message)
                        return 'ABORTED', user_feedback

                    # B. COGNITIVE PATH (Semantic): Run LLM evaluation in parallel
                    print('[Supervisor] [COGNITIVE PATH] Evaluating feedback impact semantically...')
                    decision = await ai.generate(
                        model='googleai/gemini-flash-latest',
                        system="""You are an execution supervisor. Evaluate the user's feedback against the active step.
                        Respond with exactly one word:
                        - 'ABORT': The feedback completely invalidates the current step or goal. We must stop immediately.
                        - 'COOPERATE': The feedback is valid, but the current step should finish. We will pivot the plan AFTER this step.
                        - 'QUEUE': The feedback is minor or can wait until the current plan completes.""",
                        prompt=f'Active Step: {current_step["instruction"]}\nUser Feedback: {user_feedback}',
                    )

                    strategy = decision.text.strip().upper()
                    print(f'[Supervisor] Semantic Decision: {strategy}')

                    if 'ABORT' in strategy:
                        worker_task.cancel()
                        await sess.add_messages(queued_input.message)
                        return 'ABORTED', user_feedback
                    elif 'COOPERATE' in strategy:
                        await sess.add_messages(queued_input.message)
                        return 'PIVOT_NEXT', user_feedback
                    else:
                        # Queue the feedback (append to history for later)
                        await sess.add_messages(queued_input.message)

                return 'COMPLETED', None

            monitor_task = asyncio.create_task(monitor_queue())

            # Race the worker against the supervisor monitor
            done, pending = await asyncio.wait({worker_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

            # Process the outcome of the race
            if worker_task in done:
                # The worker finished the step first
                try:
                    await worker_task

                    if feedback_event.is_set():
                        print(
                            '\n[Orchestrator] Step finished, but supervisor is evaluating feedback. Waiting for decision...'
                        )
                        await monitor_task
                        outcome, feedback = monitor_task.result()

                        if outcome == 'ABORTED':
                            await rollback_workspace(rollback_state)
                            print(f"\n[Supervisor] Recalculating plan from scratch for: '{feedback}'")
                            replanned = await ai.generate(
                                model='googleai/gemini-flash-latest',
                                system='Create a brand new step-by-step plan starting from scratch to achieve the goal.',
                                prompt=f'New Goal: {feedback}',
                            )
                            plan = parse_plan(replanned.text)
                            current_step_index = 0
                        elif outcome == 'PIVOT_NEXT':
                            plan[current_step_index]['status'] = 'completed'
                            current_step_index += 1
                            print(f"\n[Supervisor] Pivoting plan for remaining steps based on: '{feedback}'")
                            replanned = await ai.generate(
                                model='googleai/gemini-flash-latest',
                                system='Rewrite the remaining plan steps to incorporate this feedback.',
                                prompt=f'Remaining Plan: {plan[current_step_index:]}\nFeedback: {feedback}',
                            )
                            plan = plan[:current_step_index] + parse_plan(replanned.text)
                    else:
                        # No feedback received. Cancel the sleeping monitor task.
                        monitor_task.cancel()
                        plan[current_step_index]['status'] = 'completed'
                        current_step_index += 1

                    await save_state()
                    # Clear rollback tracking for this successful step
                    rollback_state['written_files'].clear()
                except asyncio.CancelledError:
                    await rollback_workspace(rollback_state)
            else:
                # The monitor task finished first (supervisor made a mid-step decision)
                outcome, feedback = await monitor_task

                if outcome == 'ABORTED':
                    # Hard cancel the active worker task immediately
                    worker_task.cancel()
                    await rollback_workspace(rollback_state)
                    print(f"\n[Supervisor] Recalculating plan from scratch for: '{feedback}'")
                    replanned = await ai.generate(
                        model='googleai/gemini-flash-latest',
                        system='Create a brand new step-by-step plan starting from scratch to achieve the goal.',
                        prompt=f'New Goal: {feedback}',
                    )
                    plan = parse_plan(replanned.text)
                    current_step_index = 0
                    await save_state()
                    print(
                        '[Supervisor] New Re-planned Plan:\n'
                        + '\n'.join([
                            f'  {i + 1}. {s["instruction"]} (Target: {s["file"]})' for i, s in enumerate(plan)
                        ])
                    )

                elif outcome == 'PIVOT_NEXT':
                    # The supervisor decided to let the active step finish, but we must pivot next.
                    # We do NOT cancel the worker. We await it to let it complete.
                    print('\n[Orchestrator] Supervisor requested PIVOT_NEXT. Letting active step finish...')
                    await worker_task
                    plan[current_step_index]['status'] = 'completed'
                    current_step_index += 1

                    print(f"\n[Supervisor] Pivoting plan for remaining steps based on: '{feedback}'")
                    replanned = await ai.generate(
                        model='googleai/gemini-flash-latest',
                        system='Rewrite the remaining plan steps to incorporate this feedback.',
                        prompt=f'Remaining Plan: {plan[current_step_index:]}\nFeedback: {feedback}',
                    )
                    plan = plan[:current_step_index] + parse_plan(replanned.text)
                    await save_state()
                    rollback_state['written_files'].clear()

        return TurnResult(finish_reason=AgentFinishReason.STOP)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='supervisedAgent', fn=supervised_agent_fn, store=store)


# --- CLI INTERACTIVE SIMULATION ---
async def interactive_client():
    print('\n=======================================================')
    print('      GENKIT DYNAMIC PREEMPTION & SUPERVISOR POC')
    print('=======================================================')
    print('Instructions:')
    print('1. Set a goal to start the agent.')
    print('2. While the worker progress bar is running, you can type:')
    print("   - 'stop' or 'abort' -> Triggers the INSTANT FAST-PATH preemption & rollback.")
    print("   - A semantic pivot (e.g., 'actually rewrite it in Python') -> Triggers COGNITIVE preemption.")
    print("   - A minor comment (e.g., 'add a docstring') -> Triggers COOPERATIVE pipeline.")
    print('=======================================================\n')

    session_id = str(uuid4())
    session = agent.chat(AgentInit(session_id=session_id))

    goal = input("Enter the agent's goal (e.g. 'Build a user authentication module'): ")
    print('\n--- Starting Session ---')

    # Send the initial goal
    turn = session.send(goal)

    # We run the turn execution in the background so we can read stdin concurrently
    async def get_output():
        return await turn.output

    turn_task = asyncio.create_task(get_output())

    loop = asyncio.get_event_loop()

    # Non-blocking stdin reader loop
    async def read_user_input():
        while not turn_task.done():
            # Run the blocking input() in a separate thread to keep the event loop moving
            user_msg = await loop.run_in_executor(None, input, '\n[You] (Type feedback mid-flight): ')
            if user_msg.strip() and not turn_task.done():
                print(f"[Client] Transmitting feedback: '{user_msg}'")
                session.send(user_msg)
                # Give a brief moment for the logs to print before displaying the prompt again
                await asyncio.sleep(0.5)

    input_task = asyncio.create_task(read_user_input())

    # Wait for the turn to complete
    await turn_task
    input_task.cancel()

    print('\n--- Session Finished ---')
    print(f'Final Session State: {session.state}')
    await session.close()


if __name__ == '__main__':
    # Ensure GEMINI_API_KEY is set
    if not os.environ.get('GEMINI_API_KEY'):
        print('Error: GEMINI_API_KEY environment variable is not set.')
        print('Please export it in your terminal:')
        print('  export GEMINI_API_KEY="your_api_key_here"')
        sys.exit(1)

    asyncio.run(interactive_client())
