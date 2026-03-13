#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Common utilities for Genkit Python samples
# ==========================================
#
# This script provides shared functions for all sample run.sh scripts.
# Source this file at the beginning of your run.sh:
#
#   source "$(dirname "$0")/../_common.sh"
#
# Available functions:
#   - print_banner "Title" "emoji"  - Print a colorful banner
#   - check_env_var "VAR_NAME" "get_url" - Check if env var is set
#   - open_browser_for_url "url" - Open browser when URL is ready
#   - genkit_start_with_browser [args...] - Start genkit and auto-open browser

# Resolve the py/ workspace root so that `samples.shared` is importable.
# _common.sh lives at py/samples/_common.sh, so py/ is one level up.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PY_ROOT="$(cd "${_COMMON_DIR}/.." && pwd)"
export PYTHONPATH="${_PY_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export CYAN='\033[0;36m'
export NC='\033[0m' # No Color

# Print a colorful banner
# Usage: print_banner "Title Text" "emoji"
print_banner() {
    local title="$1"
    local emoji="${2:-✨}"

    # Calculate padding for centering (box is 67 chars wide, content is 65)
    local content="${emoji} ${title} ${emoji}"
    local content_len=${#content}
    local padding=$(( (65 - content_len) / 2 ))
    local left_pad
    left_pad=$(printf '%*s' "$padding" '')
    local right_pad
    right_pad=$(printf '%*s' "$((65 - content_len - padding))" '')

    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    printf "║%s%s%s║\n" "$left_pad" "$content" "$right_pad"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check if an environment variable is set
# Usage: check_env_var "GOOGLE_API_KEY" "https://makersuite.google.com/app/apikey"
check_env_var() {
    local var_name="$1"
    local get_url="$2"

    local current_val="${!var_name:-}"

    # Prompt if running interactively
    # We check -t 0 (stdin is TTY) and also explicit check for /dev/tty availability
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        local display_val="${current_val}"

        # Simple masking for keys
        if [[ "$var_name" == *"API_KEY"* || "$var_name" == *"SECRET"* || "$var_name" == *"TOKEN"* ]]; then
            if [[ -n "$current_val" ]]; then
               display_val="******"
            fi
        fi

        echo -en "${BLUE}Enter ${var_name}${NC}"
        if [[ -n "$display_val" ]]; then
            echo -en " [${YELLOW}${display_val}${NC}]: "
        else
            echo -n ": "
        fi

        local input_val
        # Safely read from TTY
        if read -r input_val < /dev/tty; then
            if [[ -n "$input_val" ]]; then
                export "$var_name"="$input_val"
            fi
        fi
        # Only print newline if we actually prompted
        echo ""
    fi

    if [[ -z "${!var_name:-}" ]]; then
        echo -e "${YELLOW}Warning: ${var_name} not set${NC}"
        if [[ -n "$get_url" ]]; then
            echo "Get a key from: $get_url"
        fi
        echo ""
        return 1
    fi
    return 0
}

# Check if we have a GUI/display available
# Returns 0 (true) if GUI is available, 1 (false) otherwise
has_display() {
    # Check if running in SSH without X forwarding
    if [[ -n "${SSH_CLIENT:-}" || -n "${SSH_TTY:-}" ]]; then
        # SSH session - check for X forwarding
        if [[ -z "${DISPLAY:-}" ]]; then
            return 1  # No display in SSH without X forwarding
        fi
    fi

    # macOS always has a display if not in SSH
    if [[ "$(uname)" == "Darwin" ]]; then
        return 0
    fi

    # Linux - check for display server
    if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
        return 0
    fi

    # WSL - check for WSLg or access to Windows
    if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
        if command -v wslview &> /dev/null; then
            return 0
        fi
    fi

    # No display detected
    return 1
}

# Open browser for a given URL
# Works cross-platform: macOS, Linux, Windows (Git Bash/WSL)
# Skips browser opening if no display is available (e.g., SSH sessions)
open_browser_for_url() {
    local url="$1"

    # Check if we have a display
    if ! has_display; then
        echo -e "${CYAN}Remote session detected - skipping browser auto-open${NC}"
        echo -e "Open manually: ${GREEN}${url}${NC}"
        return 0
    fi

    if command -v open &> /dev/null; then
        open "$url"  # macOS
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$url"  # Linux
    elif command -v wslview &> /dev/null; then
        wslview "$url"  # WSL
    elif command -v start &> /dev/null; then
        start "$url"  # Windows Git Bash
    else
        echo -e "${YELLOW}Could not auto-open browser. Please open: ${GREEN}${url}${NC}"
    fi
}

# Watch genkit output for the Developer UI URL and open browser
# This function reads from stdin and watches for the URL pattern
_watch_for_devui_url() {
    local line
    local url_found=false

    while IFS= read -r line; do
        # Print the line as it comes (pass through)
        echo "$line"

        # Check for the Genkit Developer UI URL
        if [[ "$url_found" == "false" && "$line" == *"Genkit Developer UI:"* ]]; then
            # Extract URL - handle both with and without ANSI codes
            local url
            # Remove ANSI escape codes and extract URL
            url=$(echo "$line" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'https?://[^ ]+' | head -1)

            if [[ -n "$url" ]]; then
                url_found=true
                # Open browser in background
                (
                    # Small delay to ensure server is fully ready
                    sleep 1
                    open_browser_for_url "$url"
                ) &
            fi
        fi
    done
}

# Start genkit with automatic browser opening
# Usage: genkit_start_with_browser -- [your command after --]
# Example: genkit_start_with_browser -- uv run src/main.py
#
# Set GENKIT_NO_BROWSER=1 or pass --no-browser to any run.sh to
# disable automatic browser opening (useful for CI/headless).
genkit_start_with_browser() {
    # Pre-flight: ensure the genkit CLI is installed.
    # If missing, offer to install pnpm + genkit interactively.
    if ! command -v genkit &> /dev/null; then
        echo -e "${YELLOW}genkit CLI is not installed.${NC}"
        echo ""

        # Attempt interactive installation if we have a TTY.
        if [[ -t 0 ]] && [ -c /dev/tty ]; then
            # Step 1: ensure we have a JS package manager (pnpm preferred).
            if ! command -v pnpm &> /dev/null && ! command -v npm &> /dev/null; then
                if command -v node &> /dev/null && command -v corepack &> /dev/null; then
                    if _confirm "Install pnpm via corepack?"; then
                        corepack enable
                        corepack prepare pnpm@latest --activate
                    fi
                elif command -v node &> /dev/null; then
                    echo -e "${BLUE}Installing pnpm via standalone installer...${NC}"
                    curl -fsSL https://get.pnpm.io/install.sh | sh -
                    export PNPM_HOME="${PNPM_HOME:-$HOME/.local/share/pnpm}"
                    export PATH="$PNPM_HOME:$PATH"
                fi
            fi

            # Step 2: install genkit CLI.
            if command -v pnpm &> /dev/null; then
                if _confirm "Install genkit CLI via pnpm?"; then
                    pnpm install -g genkit-cli
                fi
            elif command -v npm &> /dev/null; then
                if _confirm "Install genkit CLI via npm?"; then
                    npm install -g genkit-cli
                fi
            fi
        fi

        # Final check — if genkit is still not available, exit with guidance.
        if ! command -v genkit &> /dev/null; then
            echo -e "${RED}Error: genkit CLI is required but could not be installed.${NC}"
            echo ""
            echo "The genkit CLI requires Node.js + pnpm. Install manually:"
            echo "  https://nodejs.org/           (Node.js)"
            echo "  pnpm install -g genkit-cli    (genkit CLI)"
            echo ""
            echo "Or run the setup script to install all tools:"
            echo "  ${_COMMON_DIR}/setup.sh"
            exit 1
        fi
        echo -e "${GREEN}✓ genkit CLI installed${NC}"
        echo ""
    fi

    if [[ "${GENKIT_NO_BROWSER:-}" == "1" ]]; then
        echo -e "${BLUE}Starting Genkit Dev UI (browser disabled)...${NC}"
        echo ""
        genkit start "$@"
        return
    fi

    echo -e "${BLUE}Starting Genkit Dev UI...${NC}"
    echo -e "Browser will open automatically when ready"
    echo ""

    # Run genkit start and pipe through our URL watcher
    # Using stdbuf to disable buffering for real-time output
    if command -v stdbuf &> /dev/null; then
        stdbuf -oL -eL genkit start "$@" 2>&1 | _watch_for_devui_url
    else
        # Fallback without stdbuf (may have buffering issues)
        genkit start "$@" 2>&1 | _watch_for_devui_url
    fi
}

# Prompt the user with a yes/no question. Default is yes.
# Usage: _confirm "Install uv?" && do_thing
_confirm() {
    local prompt="$1"
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "${BLUE}${prompt} [Y/n]:${NC} "
        local response
        read -r response < /dev/tty
        [[ -z "$response" || "$response" =~ ^[Yy] ]]
    else
        return 0  # Default to yes in non-interactive environments
    fi
}

# Prompt the user to run setup.sh interactively.
# Returns 0 if setup was run successfully, 1 otherwise.
# Usage: _prompt_run_setup "/path/to/setup.sh" && return
_prompt_run_setup() {
    local setup_script="$1"
    if _confirm "Run setup.sh now?"; then
        bash "$setup_script"
        return $?
    fi
    return 1
}

# Check if the development environment is set up and offer to run setup.sh.
# This is called automatically by install_deps.
check_setup() {
    local setup_script="${_COMMON_DIR}/setup.sh"

    # Auto-discover tools that setup.sh installs but may not be on PATH yet.
    # This handles the common case where the user ran setup.sh but hasn't
    # opened a new terminal or sourced ~/.environment.
    _ensure_tool_paths

    # Quick checks: is uv available? Does the workspace .venv exist?
    if ! command -v uv &>/dev/null; then
        echo -e "${YELLOW}⚠  uv is not installed.${NC}"
        echo -e "Run ${GREEN}${setup_script}${NC} to set up your development environment."
        _prompt_run_setup "$setup_script" && return
        echo -e "${RED}Cannot continue without uv. Exiting.${NC}"
        exit 1
    fi

    if [[ ! -d "${_PY_ROOT}/.venv" ]]; then
        echo -e "${YELLOW}⚠  Virtual environment not found at ${_PY_ROOT}/.venv${NC}"
        echo -e "Run ${GREEN}${setup_script}${NC} to set up your development environment."
        _prompt_run_setup "$setup_script" && return
        echo -e "${YELLOW}Continuing without setup — uv sync will create .venv...${NC}"
    fi
}

# Ensure common tool install directories are on PATH.
# setup.sh installs tools to ~/.local/bin (uv, just), npm global bins,
# and ~/.cargo/bin — but a fresh shell won't have these on PATH until
# ~/.environment (or the shell RC) is sourced.
_ensure_tool_paths() {
    # Source ~/.environment if it exists (setup.sh writes env vars there).
    local env_file="$HOME/.environment"
    if [[ -f "$env_file" ]]; then
        # shellcheck disable=SC1090
        source "$env_file"
    fi

    # Common directories where setup.sh installs tools.
    local -a extra_dirs=(
        "$HOME/.local/bin"          # uv, just, grpcurl
        "$HOME/.cargo/bin"          # older uv installs
    )

    # pnpm global bin directory (for genkit CLI installed via pnpm).
    if [[ -n "${PNPM_HOME:-}" && -d "$PNPM_HOME" ]]; then
        extra_dirs+=("$PNPM_HOME")
    elif [[ -d "$HOME/.local/share/pnpm" ]]; then
        extra_dirs+=("$HOME/.local/share/pnpm")
    fi

    # npm global bin directory (for genkit CLI installed via npm).
    if command -v npm &>/dev/null; then
        local npm_prefix
        npm_prefix="$(npm config get prefix 2>/dev/null || true)"
        if [[ -n "$npm_prefix" && -d "$npm_prefix/bin" ]]; then
            extra_dirs+=("$npm_prefix/bin")
        fi
    fi

    for dir in "${extra_dirs[@]}"; do
        if [[ -d "$dir" ]] && [[ ":$PATH:" != *":$dir:"* ]]; then
            export PATH="$dir:$PATH"
        fi
    done
}

# Install dependencies with uv
install_deps() {
    check_setup
    echo -e "${BLUE}Installing dependencies...${NC}"
    uv sync
    echo ""
}

# Standard help footer
print_help_footer() {
    local port="${1:-4000}"
    echo ""
    echo "Getting Started:"
    echo "  1. Set required environment variables"
    echo "  2. Run: ./run.sh"
    echo "  3. Browser opens automatically to http://localhost:${port}"
}

# ============================================================================
# Google Cloud (gcloud) Helper Functions
# ============================================================================
# These functions provide interactive API enablement for samples that require
# Google Cloud APIs.

# Check if gcloud CLI is installed; offer to install if missing.
# Usage: check_gcloud_installed || exit 1
check_gcloud_installed() {
    if command -v gcloud &> /dev/null; then
        echo -e "${GREEN}✓ gcloud CLI found${NC}"
        return 0
    fi

    echo -e "${YELLOW}gcloud CLI is not installed.${NC}"
    echo ""
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "Install the Google Cloud SDK now? [Y/n]: "
        local response
        read -r response < /dev/tty
        if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
            echo ""
            case "$(uname -s)" in
                Darwin)
                    if command -v brew &> /dev/null; then
                        echo -e "${BLUE}Installing via Homebrew...${NC}"
                        brew install --cask google-cloud-sdk
                    else
                        echo -e "${BLUE}Installing via curl...${NC}"
                        curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
                        # shellcheck disable=SC1091
                        source "$HOME/google-cloud-sdk/path.bash.inc" 2>/dev/null || true
                    fi
                    ;;
                Linux)
                    echo -e "${BLUE}Installing via curl...${NC}"
                    curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
                    # shellcheck disable=SC1091
                    source "$HOME/google-cloud-sdk/path.bash.inc" 2>/dev/null || true
                    ;;
                *)
                    echo "Visit: https://cloud.google.com/sdk/docs/install"
                    return 1
                    ;;
            esac
            if command -v gcloud &> /dev/null; then
                echo -e "${GREEN}✓ gcloud CLI installed successfully${NC}"
                return 0
            fi
        fi
    fi

    echo -e "${RED}Error: gcloud CLI is required${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    return 1
}

# Check if AWS CLI is installed; offer to install if missing.
# Usage: check_aws_installed || exit 1
check_aws_installed() {
    if command -v aws &> /dev/null; then
        echo -e "${GREEN}✓ AWS CLI found${NC}"
        return 0
    fi

    echo -e "${YELLOW}AWS CLI is not installed.${NC}"
    echo ""
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "Install the AWS CLI now? [Y/n]: "
        local response
        read -r response < /dev/tty
        if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
            echo ""
            case "$(uname -s)" in
                Darwin)
                    if command -v brew &> /dev/null; then
                        echo -e "${BLUE}Installing via Homebrew...${NC}"
                        brew install awscli
                    else
                        echo -e "${BLUE}Installing via pkg...${NC}"
                        curl -fsSL "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o /tmp/AWSCLIV2.pkg
                        sudo installer -pkg /tmp/AWSCLIV2.pkg -target /
                        rm -f /tmp/AWSCLIV2.pkg
                    fi
                    ;;
                Linux)
                    echo -e "${BLUE}Installing AWS CLI v2...${NC}"
                    # The AWS CLI zip installer requires unzip.
                    if ! command -v unzip &> /dev/null; then
                        echo -e "${YELLOW}unzip is required but not installed. Attempting to install...${NC}"
                        if command -v apt-get &> /dev/null; then
                            sudo apt-get update -qq && sudo apt-get install -yqq unzip
                        elif command -v dnf &> /dev/null; then
                            sudo dnf install -yq unzip
                        elif command -v yum &> /dev/null; then
                            sudo yum install -yq unzip
                        else
                            echo -e "${RED}Error: unzip is required. Install it manually and retry.${NC}"
                            return 1
                        fi
                    fi
                    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
                    unzip -qo /tmp/awscliv2.zip -d /tmp
                    sudo /tmp/aws/install || /tmp/aws/install --install-dir "$HOME/.local/aws-cli" --bin-dir "$HOME/.local/bin"
                    rm -rf /tmp/awscliv2.zip /tmp/aws
                    ;;
                *)
                    echo "Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
                    return 1
                    ;;
            esac
            if command -v aws &> /dev/null; then
                echo -e "${GREEN}✓ AWS CLI installed successfully${NC}"
                return 0
            fi
        fi
    fi

    echo -e "${RED}Error: AWS CLI is required${NC}"
    echo "Install from: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    return 1
}

# Check if Azure CLI is installed; offer to install if missing.
# Usage: check_az_installed || exit 1
check_az_installed() {
    if command -v az &> /dev/null; then
        echo -e "${GREEN}✓ Azure CLI found${NC}"
        return 0
    fi

    echo -e "${YELLOW}Azure CLI is not installed.${NC}"
    echo ""
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "Install the Azure CLI now? [Y/n]: "
        local response
        read -r response < /dev/tty
        if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
            echo ""
            case "$(uname -s)" in
                Darwin)
                    if command -v brew &> /dev/null; then
                        echo -e "${BLUE}Installing via Homebrew...${NC}"
                        brew install azure-cli
                    else
                        echo -e "${BLUE}Installing via script...${NC}"
                        curl -fsSL https://aka.ms/InstallAzureCLIDeb | bash
                    fi
                    ;;
                Linux)
                    # Detect distro and use the appropriate Azure CLI install.
                    # https://learn.microsoft.com/cli/azure/install-azure-cli-linux
                    if [ -f /etc/os-release ]; then
                        # shellcheck disable=SC1091
                        . /etc/os-release
                    fi
                    case "${ID:-}" in
                        debian|ubuntu|linuxmint|pop)
                            echo -e "${BLUE}Installing via InstallAzureCLIDeb...${NC}"
                            curl -fsSL https://aka.ms/InstallAzureCLIDeb | sudo bash
                            ;;
                        fedora|rhel|centos|rocky|alma)
                            echo -e "${BLUE}Installing via dnf/yum...${NC}"
                            sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
                            sudo dnf install -y https://packages.microsoft.com/config/rhel/9.0/packages-microsoft-prod.rpm \
                                || sudo yum install -y https://packages.microsoft.com/config/rhel/9.0/packages-microsoft-prod.rpm
                            sudo dnf install -y azure-cli || sudo yum install -y azure-cli
                            ;;
                        *)
                            echo -e "${BLUE}Installing via pip (distro '${ID:-unknown}' not directly supported)...${NC}"
                            pip install azure-cli
                            ;;
                    esac
                    ;;
                *)
                    echo "Visit: https://learn.microsoft.com/cli/azure/install-azure-cli"
                    return 1
                    ;;
            esac
            if command -v az &> /dev/null; then
                echo -e "${GREEN}✓ Azure CLI installed successfully${NC}"
                return 0
            fi
        fi
    fi

    echo -e "${RED}Error: Azure CLI is required${NC}"
    echo "Install from: https://learn.microsoft.com/cli/azure/install-azure-cli"
    return 1
}

# Check if Ollama is installed; offer to install if missing.
# Also ensures the Ollama server is running.
# Usage: check_ollama_installed || exit 1
check_ollama_installed() {
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓ Ollama found${NC}"
    else
        echo -e "${YELLOW}Ollama is not installed.${NC}"
        echo ""
        if [[ -t 0 ]] && [ -c /dev/tty ]; then
            echo -en "Install Ollama now? [Y/n]: "
            local response
            read -r response < /dev/tty
            if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
                echo ""
                case "$(uname -s)" in
                    Darwin)
                        if command -v brew &> /dev/null; then
                            echo -e "${BLUE}Installing via Homebrew...${NC}"
                            brew install ollama
                        else
                            echo -e "${BLUE}Installing via official script...${NC}"
                            curl -fsSL https://ollama.com/install.sh | sh
                        fi
                        ;;
                    Linux)
                        # Prefer system package manager when available.
                        local installed_via_pkg=false
                        if command -v apt-get &>/dev/null; then
                            if apt-cache show ollama &>/dev/null 2>&1; then
                                echo -e "${BLUE}Installing via apt...${NC}"
                                sudo apt-get update -qq
                                sudo apt-get install -y -qq ollama
                                installed_via_pkg=true
                            fi
                        elif command -v dnf &>/dev/null; then
                            if dnf info ollama &>/dev/null 2>&1; then
                                echo -e "${BLUE}Installing via dnf...${NC}"
                                sudo dnf install -y -q ollama
                                installed_via_pkg=true
                            fi
                        fi
                        if ! $installed_via_pkg; then
                            echo -e "${BLUE}Installing via official script...${NC}"
                            curl -fsSL https://ollama.com/install.sh | sh
                        fi
                        ;;
                    *)
                        echo "Visit: https://ollama.com/download"
                        return 1
                        ;;
                esac
                if command -v ollama &> /dev/null; then
                    echo -e "${GREEN}✓ Ollama installed successfully${NC}"
                else
                    echo -e "${RED}Error: Ollama installation failed${NC}"
                    return 1
                fi
            else
                echo -e "${RED}Error: Ollama is required${NC}"
                echo "Install from: https://ollama.com/download"
                return 1
            fi
        else
            echo -e "${RED}Error: Ollama is required${NC}"
            echo "Install from: https://ollama.com/download"
            return 1
        fi
    fi

    # Ensure the Ollama server is running
    if curl -sf http://localhost:11434/api/tags &> /dev/null; then
        echo -e "${GREEN}✓ Ollama server is running${NC}"
    else
        echo -e "${YELLOW}Ollama server not responding, starting it...${NC}"
        ollama serve &> /dev/null &
        # Wait up to 10 seconds for the server to start
        local retries=20
        while (( retries > 0 )); do
            if curl -sf http://localhost:11434/api/tags &> /dev/null; then
                echo -e "${GREEN}✓ Ollama server started${NC}"
                break
            fi
            sleep 0.5
            retries=$((retries - 1))
        done
        if (( retries == 0 )); then
            echo -e "${YELLOW}⚠ Ollama server may not have started. Try 'ollama serve' manually.${NC}"
        fi
    fi

    return 0
}

# Check if flyctl CLI is installed; offer to install if missing.
# Usage: check_flyctl_installed || exit 1
check_flyctl_installed() {
    if command -v flyctl &> /dev/null; then
        echo -e "${GREEN}✓ flyctl CLI found${NC}"
        return 0
    fi

    echo -e "${YELLOW}flyctl CLI is not installed.${NC}"
    echo ""
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "Install flyctl now? [Y/n]: "
        local response
        read -r response < /dev/tty
        if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
            echo ""
            echo -e "${BLUE}Installing flyctl...${NC}"
            curl -fsSL https://fly.io/install.sh | sh
            export PATH="$HOME/.fly/bin:$PATH"
            if command -v flyctl &> /dev/null; then
                echo -e "${GREEN}✓ flyctl installed successfully${NC}"
                return 0
            fi
        fi
    fi

    echo -e "${RED}Error: flyctl is required${NC}"
    echo "Install from: https://fly.io/docs/flyctl/install/"
    return 1
}

# Check if gcloud is authenticated with Application Default Credentials.
# Prompts the user to login if not authenticated (interactive).
# Usage: check_gcloud_auth || true
check_gcloud_auth() {
    echo -e "${BLUE}Checking gcloud authentication...${NC}"

    # Check application default credentials
    if ! gcloud auth application-default print-access-token &> /dev/null; then
        echo -e "${YELLOW}Application default credentials not found.${NC}"
        echo ""

        if [[ -t 0 ]] && [ -c /dev/tty ]; then
            echo -en "Run ${GREEN}gcloud auth application-default login${NC} now? [Y/n]: "
            local response
            read -r response < /dev/tty
            if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
                echo ""
                gcloud auth application-default login
                echo ""
            else
                echo -e "${YELLOW}Skipping authentication. You may encounter auth errors.${NC}"
                return 1
            fi
        else
            echo "Run: gcloud auth application-default login"
            return 1
        fi
    else
        echo -e "${GREEN}✓ Application default credentials found${NC}"
    fi

    echo ""
    return 0
}

# Check if AWS CLI is authenticated.
# Prompts the user to run `aws configure` if no credentials found.
# Usage: check_aws_auth || true
check_aws_auth() {
    echo -e "${BLUE}Checking AWS authentication...${NC}"

    if aws sts get-caller-identity &> /dev/null; then
        echo -e "${GREEN}✓ AWS credentials found${NC}"
        echo ""
        return 0
    fi

    echo -e "${YELLOW}AWS credentials not found.${NC}"
    echo ""

    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "Run ${GREEN}aws configure${NC} now? [Y/n]: "
        local response
        read -r response < /dev/tty
        if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
            echo ""
            aws configure
            echo ""
        else
            echo -e "${YELLOW}Skipping authentication. You may encounter auth errors.${NC}"
            return 1
        fi
    else
        echo "Run: aws configure"
        return 1
    fi

    return 0
}

# Check if Azure CLI is authenticated.
# Prompts the user to run `az login` if no credentials found.
# Usage: check_az_auth || true
check_az_auth() {
    echo -e "${BLUE}Checking Azure authentication...${NC}"

    if az account show &> /dev/null; then
        echo -e "${GREEN}✓ Azure credentials found${NC}"
        echo ""
        return 0
    fi

    echo -e "${YELLOW}Azure credentials not found.${NC}"
    echo ""

    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -en "Run ${GREEN}az login${NC} now? [Y/n]: "
        local response
        read -r response < /dev/tty
        if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
            echo ""
            az login
            echo ""
        else
            echo -e "${YELLOW}Skipping authentication. You may encounter auth errors.${NC}"
            return 1
        fi
    else
        echo "Run: az login"
        return 1
    fi

    return 0
}

# Check if a specific Google Cloud API is enabled
# Usage: is_api_enabled "aiplatform.googleapis.com" "$GOOGLE_CLOUD_PROJECT"
is_api_enabled() {
    local api="$1"
    local project="$2"

    gcloud services list --project="$project" --enabled --filter="name:$api" --format="value(name)" 2>/dev/null | grep -q "$api"
}

# Enable required Google Cloud APIs interactively
# Usage:
#   REQUIRED_APIS=("aiplatform.googleapis.com" "discoveryengine.googleapis.com")
#   enable_required_apis "${REQUIRED_APIS[@]}"
#
# The function will:
#   1. Check which APIs are already enabled
#   2. Prompt the user to enable missing APIs
#   3. Enable APIs on user confirmation
enable_required_apis() {
    local project="${GOOGLE_CLOUD_PROJECT:-}"
    local apis=("$@")

    if [[ -z "$project" ]]; then
        echo -e "${YELLOW}GOOGLE_CLOUD_PROJECT not set, skipping API enablement${NC}"
        return 1
    fi

    if [[ ${#apis[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No APIs specified${NC}"
        return 0
    fi

    echo -e "${BLUE}Checking required APIs for project: ${project}${NC}"

    local apis_to_enable=()

    for api in "${apis[@]}"; do
        if is_api_enabled "$api" "$project"; then
            echo -e "  ${GREEN}✓${NC} $api"
        else
            echo -e "  ${YELLOW}✗${NC} $api (not enabled)"
            apis_to_enable+=("$api")
        fi
    done

    echo ""

    if [[ ${#apis_to_enable[@]} -eq 0 ]]; then
        echo -e "${GREEN}All required APIs are already enabled!${NC}"
        echo ""
        return 0
    fi

    # Prompt to enable APIs
    if [[ -t 0 ]] && [ -c /dev/tty ]; then
        echo -e "${YELLOW}The following APIs need to be enabled:${NC}"
        for api in "${apis_to_enable[@]}"; do
            echo "  - $api"
        done
        echo ""
        echo -en "Enable these APIs now? [Y/n]: "
        local response
        read -r response < /dev/tty

        if [[ -z "$response" || "$response" =~ ^[Yy] ]]; then
            echo ""
            for api in "${apis_to_enable[@]}"; do
                echo -e "${BLUE}Enabling $api...${NC}"
                if gcloud services enable "$api" --project="$project"; then
                    echo -e "${GREEN}✓ Enabled $api${NC}"
                else
                    echo -e "${RED}✗ Failed to enable $api${NC}"
                    return 1
                fi
            done
            echo ""
            echo -e "${GREEN}All APIs enabled successfully!${NC}"
        else
            echo -e "${YELLOW}Skipping API enablement. You may encounter errors.${NC}"
            return 1
        fi
    else
        echo "Enable APIs with:"
        for api in "${apis_to_enable[@]}"; do
            echo "  gcloud services enable $api --project=$project"
        done
        return 1
    fi

    echo ""
    return 0
}

# Run common GCP setup: check gcloud, auth, and enable APIs
# Usage:
#   REQUIRED_APIS=("aiplatform.googleapis.com")
#   run_gcp_setup "${REQUIRED_APIS[@]}"
run_gcp_setup() {
    local apis=("$@")

    # Check gcloud is installed
    check_gcloud_installed || return 1

    # Check/prompt for project
    check_env_var "GOOGLE_CLOUD_PROJECT" "" || {
        echo -e "${RED}Error: GOOGLE_CLOUD_PROJECT is required${NC}"
        echo ""
        echo "Set it with:"
        echo "  export GOOGLE_CLOUD_PROJECT=your-project-id"
        echo ""
        return 1
    }

    # Check authentication
    check_gcloud_auth || true

    # Enable APIs if any were specified
    if [[ ${#apis[@]} -gt 0 ]]; then
        enable_required_apis "${apis[@]}" || true
    fi

    return 0
}
