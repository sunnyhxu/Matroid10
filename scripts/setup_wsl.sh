#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  nauty \
  libnauty-dev \
  python3 \
  python3-pip \
  curl \
  bzip2

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if apt-cache policy sagemath 2>/dev/null | grep -q "Candidate: (none)"; then
  echo "APT package 'sagemath' is unavailable on this Ubuntu release. Installing Sage via micromamba (conda-forge)."

  mkdir -p "$HOME/.local/bin"
  if ! command -v micromamba >/dev/null 2>&1; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "$HOME/.local/bin" --strip-components=1 bin/micromamba
  fi

  if ! "$HOME/.local/bin/micromamba" env list | awk '{print $1}' | grep -qx "sage"; then
    "$HOME/.local/bin/micromamba" create -y -n sage -c conda-forge sage python=3.12
  fi

  cat > "$HOME/.local/bin/sage" <<'EOF'
#!/usr/bin/env bash
exec "$HOME/.local/bin/micromamba" run -n sage sage "$@"
EOF
  chmod +x "$HOME/.local/bin/sage"
else
  sudo apt-get install -y sagemath
fi

echo "Setup complete."
echo "If needed, add ~/.local/bin to PATH:"
echo "  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && source ~/.bashrc"
