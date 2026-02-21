# Runs bootstrap via docker exec (garage image is distroless, no shell; -h requires node-id).
FROM alpine:3.20

RUN apk add --no-cache docker-cli

COPY scripts/garage_bootstrap.sh /bootstrap.sh
RUN chmod +x /bootstrap.sh

ENV GARAGE_CONTAINER=garage
ENTRYPOINT ["/bootstrap.sh"]
