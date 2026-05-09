/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { EXAMPLE_EVENTS, EXAMPLE_GRADES, getUpcomingHolidays } from './data.js';
import { ai, z } from './genkit.js';
import type { AgentState } from './types.js';

export const searchEvents = ai.defineTool(
  {
    name: 'searchEvents',
    description:
      'use this when asked about any time/location for school events including extra curriculars like clubs',
    inputSchema: z.object({
      activity: z
        .string()
        .optional()
        .describe(
          'if looking for a particular activity, provide it here. must be an exact match for an activity name'
        ),
      grade: z
        .number()
        .optional()
        .describe('restrict searched events to a particular grade level'),
    }),
  },
  async ({ activity, grade }) => {
    return EXAMPLE_EVENTS.filter(
      (e) => !grade || e.grades.includes(grade)
    ).filter(
      (e) => !activity || e.activity?.toLowerCase() === activity?.toLowerCase()
    );
  }
);

function checkIsParent(studentId: number, state: AgentState) {
  const student = state.students.find((s) => s.id === studentId);
  if (!student) {
    throw new Error(
      `Unable to process request for student ID ${studentId}. Parents can only submit requests for their registered children.`
    );
  }
  return student;
}

export const reportAbsence = ai.defineTool(
  {
    name: 'reportAbsence',
    description:
      'use this tool to mark a specific student as absent on one or more days',
    inputSchema: z.object({
      studentId: z.number().describe('the id of the student'),
      date: z.string().describe('the date of the absence in YYYY-MM-DD format'),
      reason: z.string().describe('the provided reason for the absence'),
      excused: z
        .boolean()
        .describe('whether the absence is excused by the parent'),
    }),
  },
  async (input) => {
    const student = checkIsParent(
      input.studentId,
      ai.currentSession<AgentState>().state!
    );
    console.log(
      `[TOOL] Absence reported for ${student.name} (ID: ${input.studentId}) on ${input.date}`
    );
    return { ok: true, message: 'Absence successfully recorded' };
  }
);

export const reportTardy = ai.defineTool(
  {
    name: 'reportTardy',
    description:
      'use this tool to mark a specific student tardy for a given date',
    inputSchema: z.object({
      studentId: z.number().describe('the id of the student'),
      date: z.string().describe('the date of the tardy'),
      reason: z.string().describe('the provided reason reason for the tardy'),
      eta: z
        .string()
        .describe(
          'the time the student is expected to arrive at school in HH:MMam/pm format'
        ),
      excused: z
        .boolean()
        .describe('whether the absence is excused by the parent'),
    }),
  },
  async (input) => {
    checkIsParent(input.studentId, ai.currentSession<AgentState>().state!);
    console.log(
      '[TOOL] Student',
      input.studentId,
      'has been reported tardy for',
      input.date
    );
    return { ok: true };
  }
);

export const upcomingHolidays = ai.defineTool(
  {
    name: 'upcomingHolidays',
    description: 'can retrieve information about upcoming holidays',
    outputSchema: z.string(),
  },
  async () => JSON.stringify(await getUpcomingHolidays())
);

export const getRecentGrades = ai.defineTool(
  {
    name: 'getRecentGrades',
    description: 'retrieves recent grades for a specific student',
    inputSchema: z.object({
      studentId: z.number().describe('the id of the student'),
      subject: z.string().optional().describe('optional subject filter'),
      limit: z
        .number()
        .optional()
        .describe('number of recent grades to return'),
    }),
  },
  async ({ studentId, subject, limit = 5 }) => {
    checkIsParent(studentId, ai.currentSession<AgentState>().state!);
    let grades = EXAMPLE_GRADES.filter((g) => g.studentId === studentId);
    if (subject) {
      grades = grades.filter(
        (g) => g.subject.toLowerCase() === subject.toLowerCase()
      );
    }
    return grades.slice(0, limit);
  }
);
