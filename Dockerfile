# mcr.microsoft.com/devcontainers/python already ships a non-root "vscode" user with zsh + oh-my-zsh
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# base image ships an unusable dl.yarnpkg.com apt source (missing GPG key); drop it before updating
RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/*yarn*.list

# build-essential/gfortran cover platforms where numpy/scipy/obspy fall back to source builds
# cache mount keeps downloaded .debs across rebuilds instead of re-fetching every time
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential gfortran

# vendor uv from its official image instead of piping an install script through shell
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
