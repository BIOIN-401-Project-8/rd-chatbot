FROM rapiz1/rathole:v0.5.0 as rathole

FROM debian:12.5
WORKDIR /app
COPY --from=rathole /app/rathole .
COPY --from=rathole /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
