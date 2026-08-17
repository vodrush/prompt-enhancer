#!/bin/sh
# OmniRoute misdetects the container as Android/Termux without a writable cache
# dir and its Next.js instrumentation fails -> /v1/models returns 500.
mkdir -p /root/.cache
exec omniroute serve --port 20128
