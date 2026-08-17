# OmniRoute gateway — same npm package as the native Windows install.
# The existing config/data (providers, combos, API keys, SQLite) is bind-mounted
# from ~/.omniroute, so nothing is reconfigured.
FROM node:22-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 make g++ git \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g omniroute@3.8.49

EXPOSE 20128
COPY docker/omniroute/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
