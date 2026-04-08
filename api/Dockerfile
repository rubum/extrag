FROM rust:slim-bookworm AS builder

# Install dependencies needed for sys crates / cc
RUN apt-get update && apt-get install -y pkg-config libssl-dev build-essential

WORKDIR /usr/src/extrag

# Copy the entire workspace
COPY . .

# Build the API binary
RUN cargo build --release --package api

# Create the final lean image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built binary
COPY --from=builder /usr/src/extrag/target/release/api /app/api

# Copy the web directory so ServeDir can host it natively
COPY --from=builder /usr/src/extrag/web /app/web

# Expose the API port
EXPOSE 8080

CMD ["/app/api"]
