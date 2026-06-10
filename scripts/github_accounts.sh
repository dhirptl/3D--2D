#!/usr/bin/env bash
# GitHub CLI account helpers (work vs personal).
# One-time setup (run in your terminal, complete browser prompts):
#   gh auth login -h github.com -p https -w    # personal: dhirptl
#   gh auth login -h github.com -p https -w    # work: dhirp-clutchvr
#   gh auth setup-git
#
# Usage after both accounts are logged in:
#   source scripts/github_accounts.sh
#   gh-personal && git push
#   gh-work    && git push

gh-personal() {
  gh auth switch -h github.com -u dhirptl
}

gh-work() {
  gh auth switch -h github.com -u dhirp-clutchvr
}

gh-whoami() {
  gh auth status
}
