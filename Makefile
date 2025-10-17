# Hauptbuch Makefile
# Common development tasks

.PHONY: help build test bench fmt clippy audit clean dev docker stop logs monitor install setup

# Default target
help:
	@echo "Hauptbuch Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  setup     - Initial project setup"
	@echo "  install   - Install dependencies"
	@echo ""
	@echo "Development:"
	@echo "  build     - Build the project"
	@echo "  test      - Run tests"
	@echo "  bench     - Run benchmarks"
	@echo "  fmt       - Format code"
	@echo "  clippy    - Run clippy linter"
	@echo "  audit     - Security audit"
	@echo "  clean     - Clean build artifacts"
	@echo "  dev       - Start development server"
	@echo ""
	@echo "Docker:"
	@echo "  docker    - Start Docker services"
	@echo "  stop      - Stop Docker services"
	@echo "  logs      - Show Docker logs"
	@echo "  monitor   - Open monitoring dashboard"
	@echo ""
	@echo "All-in-one:"
	@echo "  all       - Run build, test, fmt, clippy, audit"

# Setup
setup:
	@echo "🚀 Setting up Hauptbuch..."
	@./scripts/setup.sh

install:
	@echo "📦 Installing dependencies..."
	@cargo build

# Development
build:
	@echo "🔨 Building Hauptbuch..."
	@cargo build

test:
	@echo "🧪 Running tests..."
	@cargo test

bench:
	@echo "📊 Running benchmarks..."
	@cargo bench

fmt:
	@echo "🎨 Formatting code..."
	@cargo fmt

clippy:
	@echo "🔍 Running clippy..."
	@cargo clippy -- -D warnings

audit:
	@echo "🔒 Running security audit..."
	@cargo audit

clean:
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean

dev:
	@echo "🚀 Starting development server..."
	@cargo watch -x run

# Docker
docker:
	@echo "🐳 Starting Docker services..."
	@docker-compose up -d

stop:
	@echo "🛑 Stopping Docker services..."
	@docker-compose down

logs:
	@echo "📋 Showing Docker logs..."
	@docker-compose logs -f

monitor:
	@echo "📊 Opening monitoring dashboard..."
	@if command -v xdg-open > /dev/null; then xdg-open http://localhost:3000; \
	elif command -v open > /dev/null; then open http://localhost:3000; \
	else echo "Please open http://localhost:3000 in your browser"; fi

# All-in-one
all: build test fmt clippy audit
	@echo "✅ All checks completed!"

# Quick development cycle
quick: fmt clippy test
	@echo "⚡ Quick development cycle completed!"

# Full development cycle
full: clean build test bench fmt clippy audit
	@echo "🎯 Full development cycle completed!"

# Release build
release:
	@echo "🚀 Building release..."
	@cargo build --release

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@cargo doc --open

# Security
security: audit clippy
	@echo "🔒 Security checks completed!"

# Performance
perf: bench
	@echo "⚡ Performance benchmarks completed!"

# CI/CD
ci: build test fmt clippy audit
	@echo "🔄 CI pipeline completed!"

# Production
prod: clean release test
	@echo "🏭 Production build completed!"
