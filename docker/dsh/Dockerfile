# DeepSeek Harness (dsh) — Web UI containerized so it boots with the stack.
FROM node:24-slim

# node-pty + koffi compilent au postinstall : build-essential + python requis
# socat : pont 0.0.0.0 -> 127.0.0.1 car dsh refuse --host 0.0.0.0 (RCE guard)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 make g++ git socat \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @deepseek-ai/dsh --allow-scripts=@deepseek-ai/dsh-subprocess-local,koffi,node-pty,@google/genai,protobufjs

ENV DSH_HOME=/dsh-home
EXPOSE 3080

# Les bind mounts Windows exposent les fichiers en mode 777, ce que
# dsh-credentials-local refuse (owner-only). On resserre les permissions
# avant chaque boot, puis on bridge 3080 externe -> 127.0.0.1:3080 (dsh).
CMD ["sh", "-c", "chmod 600 /dsh-home/.credentials.yaml 2>/dev/null; chmod 700 /dsh-home 2>/dev/null; socat TCP-LISTEN:3080,fork,reuseaddr TCP:127.0.0.1:3081 & exec dsh web --port 3081"]
