# Contributing Guide

## Overview

Thank you for your interest in contributing to Hauptbuch! This guide provides comprehensive instructions for contributing to the project, including development setup, coding standards, and submission process.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Contribution Process](#contribution-process)
- [Code Review](#code-review)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)
- [Release Process](#release-process)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- **Rust**: Version 1.70 or later
- **Git**: For version control
- **Docker**: For containerized development
- **IDE**: VS Code, IntelliJ, or your preferred editor

### Development Environment

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/hauptbuch.git
   cd hauptbuch
   ```

2. **Set Up Remote**
   ```bash
   # Add upstream remote
   git remote add upstream https://github.com/hauptbuch/hauptbuch.git
   
   # Verify remotes
   git remote -v
   ```

3. **Install Dependencies**
   ```bash
   # Install system dependencies
   ./scripts/install-deps.sh
   
   # Install Rust dependencies
   cargo build
   ```

## Development Setup

### Project Structure

```
hauptbuch/
├── src/                    # Source code
│   ├── consensus/          # Consensus modules
│   ├── crypto/             # Cryptography modules
│   ├── network/            # Network modules
│   ├── database/           # Database modules
│   └── ...
├── tests/                  # Test files
├── benchmarks/             # Benchmark files
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── examples/               # Example code
```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   # Create and switch to feature branch
   git checkout -b feature/your-feature-name
   
   # Or for bug fixes
   git checkout -b bugfix/issue-description
   ```

2. **Make Changes**
   ```bash
   # Make your changes
   # Follow coding standards
   # Add tests
   # Update documentation
   ```

3. **Test Changes**
   ```bash
   # Run tests
   cargo test
   
   # Run benchmarks
   cargo bench
   
   # Run linting
   cargo clippy
   
   # Format code
   cargo fmt
   ```

4. **Commit Changes**
   ```bash
   # Stage changes
   git add .
   
   # Commit with descriptive message
   git commit -m "feat: add new feature description"
   ```

5. **Push Changes**
   ```bash
   # Push to your fork
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill out the template
   - Submit for review

## Coding Standards

### Rust Style Guide

1. **Code Formatting**
   ```bash
   # Use rustfmt for formatting
   cargo fmt
   
   # Use clippy for linting
   cargo clippy
   ```

2. **Naming Conventions**
   ```rust
   // Use snake_case for variables and functions
   let user_name = "john_doe";
   fn calculate_hash() -> String { }
   
   // Use PascalCase for types and traits
   struct UserAccount { }
   trait DatabaseInterface { }
   
   // Use SCREAMING_SNAKE_CASE for constants
   const MAX_CONNECTIONS: usize = 100;
   ```

3. **Documentation**
   ```rust
   /// Calculate the hash of a block
   /// 
   /// # Arguments
   /// 
   /// * `block` - The block to hash
   /// 
   /// # Returns
   /// 
   /// * `String` - The hexadecimal hash of the block
   /// 
   /// # Examples
   /// 
   /// ```
   /// let block = Block::new();
   /// let hash = calculate_block_hash(&block);
   /// assert!(!hash.is_empty());
   /// ```
   pub fn calculate_block_hash(block: &Block) -> String {
       // Implementation
   }
   ```

4. **Error Handling**
   ```rust
   use std::error::Error;
   
   #[derive(Debug)]
   pub enum ConsensusError {
       InvalidBlock,
       InvalidTransaction,
       NetworkError(String),
   }
   
   impl Error for ConsensusError { }
   
   impl std::fmt::Display for ConsensusError {
       fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
           match self {
               ConsensusError::InvalidBlock => write!(f, "Invalid block"),
               ConsensusError::InvalidTransaction => write!(f, "Invalid transaction"),
               ConsensusError::NetworkError(msg) => write!(f, "Network error: {}", msg),
           }
       }
   }
   ```

### Code Quality

1. **Code Review Checklist**
   - [ ] Code follows style guidelines
   - [ ] Functions are well-documented
   - [ ] Error handling is appropriate
   - [ ] Tests are included
   - [ ] Performance is considered
   - [ ] Security is addressed

2. **Performance Considerations**
   ```rust
   // Use efficient data structures
   use std::collections::HashMap;
   
   // Avoid unnecessary allocations
   let mut buffer = String::with_capacity(1024);
   
   // Use appropriate concurrency
   use tokio::sync::Mutex;
   let shared_data = Arc::new(Mutex::new(data));
   ```

3. **Security Considerations**
   ```rust
   // Use secure random number generation
   use rand::rngs::OsRng;
   let mut rng = OsRng;
   
   // Validate all inputs
   pub fn validate_transaction(tx: &Transaction) -> Result<(), ValidationError> {
       if tx.value() < 0 {
           return Err(ValidationError::InvalidValue);
       }
       // More validation...
   }
   ```

## Contribution Process

### Issue Types

1. **Bug Reports**
   - Use the bug report template
   - Provide reproduction steps
   - Include system information
   - Attach relevant logs

2. **Feature Requests**
   - Use the feature request template
   - Describe the use case
   - Provide implementation ideas
   - Consider backward compatibility

3. **Documentation**
   - Fix typos and errors
   - Improve clarity
   - Add missing information
   - Update examples

### Pull Request Process

1. **Before Submitting**
   ```bash
   # Ensure all tests pass
   cargo test
   
   # Run benchmarks
   cargo bench
   
   # Check formatting
   cargo fmt --check
   
   # Run linting
   cargo clippy
   ```

2. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

3. **Review Process**
   - Maintainers review within 48 hours
   - Address feedback promptly
   - Make requested changes
   - Respond to questions

## Code Review

### Review Guidelines

1. **For Contributors**
   - Respond to feedback quickly
   - Make requested changes
   - Ask questions if unclear
   - Be respectful and constructive

2. **For Reviewers**
   - Be constructive and helpful
   - Focus on code quality
   - Check for security issues
   - Verify test coverage

3. **Review Criteria**
   - Code correctness
   - Performance implications
   - Security considerations
   - Documentation quality
   - Test coverage

### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs
   - Tests must pass
   - Code coverage checked
   - Security scans run

2. **Manual Review**
   - Code quality assessment
   - Architecture review
   - Performance analysis
   - Security review

3. **Approval Process**
   - At least one approval required
   - Maintainer approval for major changes
   - Security team approval for security changes

## Testing

### Test Types

1. **Unit Tests**
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_block_creation() {
           let block = Block::new();
           assert!(block.is_valid());
       }
   }
   ```

2. **Integration Tests**
   ```rust
   #[tokio::test]
   async fn test_consensus_integration() {
       let mut consensus = ConsensusEngine::new();
       let block = consensus.create_block(vec![]);
       assert!(block.is_valid());
   }
   ```

3. **Performance Tests**
   ```rust
   use criterion::{black_box, criterion_group, criterion_main, Criterion};
   
   fn benchmark_block_creation(c: &mut Criterion) {
       c.bench_function("block_creation", |b| {
           b.iter(|| {
               let block = Block::new();
               black_box(block)
           })
       });
   }
   ```

### Test Requirements

1. **Coverage**
   - Aim for 80%+ code coverage
   - Test edge cases
   - Test error conditions
   - Test performance

2. **Quality**
   - Tests should be reliable
   - Tests should be fast
   - Tests should be maintainable
   - Tests should be documented

## Documentation

### Documentation Standards

1. **Code Documentation**
   ```rust
   /// Calculate the hash of a block using SHA-256
   /// 
   /// # Arguments
   /// 
   /// * `block` - The block to hash
   /// 
   /// # Returns
   /// 
   /// * `String` - The hexadecimal hash of the block
   /// 
   /// # Examples
   /// 
   /// ```
   /// let block = Block::new();
   /// let hash = calculate_block_hash(&block);
   /// assert!(!hash.is_empty());
   /// ```
   pub fn calculate_block_hash(block: &Block) -> String {
       // Implementation
   }
   ```

2. **API Documentation**
   - Document all public APIs
   - Include examples
   - Explain parameters
   - Document return values

3. **User Documentation**
   - Keep documentation up-to-date
   - Use clear language
   - Include examples
   - Provide troubleshooting

### Documentation Process

1. **Before Changes**
   - Read existing documentation
   - Understand the context
   - Plan documentation updates

2. **During Development**
   - Document as you code
   - Update examples
   - Add inline comments

3. **After Changes**
   - Review documentation
   - Update examples
   - Test documentation

## Community Guidelines

### Code of Conduct

1. **Be Respectful**
   - Treat everyone with respect
   - Use inclusive language
   - Be constructive in feedback

2. **Be Professional**
   - Focus on technical issues
   - Avoid personal attacks
   - Maintain confidentiality

3. **Be Collaborative**
   - Help others learn
   - Share knowledge
   - Work together

### Communication

1. **GitHub Issues**
   - Use for bug reports
   - Use for feature requests
   - Use for discussions

2. **Discord**
   - Use for real-time chat
   - Use for quick questions
   - Use for community discussions

3. **Email**
   - Use for security issues
   - Use for sensitive matters
   - Use for formal communication

## Release Process

### Release Types

1. **Major Releases**
   - Breaking changes
   - New features
   - Architecture changes

2. **Minor Releases**
   - New features
   - Bug fixes
   - Performance improvements

3. **Patch Releases**
   - Bug fixes
   - Security updates
   - Documentation updates

### Release Process

1. **Preparation**
   - Update version numbers
   - Update changelog
   - Run full test suite
   - Create release notes

2. **Release**
   - Create release branch
   - Tag release
   - Build artifacts
   - Publish release

3. **Post-Release**
   - Monitor for issues
   - Update documentation
   - Communicate changes

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Clean build cache
   cargo clean
   
   # Update dependencies
   cargo update
   
   # Rebuild
   cargo build
   ```

2. **Test Failures**
   ```bash
   # Run specific test
   cargo test test_name
   
   # Run with debug output
   RUST_LOG=debug cargo test
   
   # Run tests in parallel
   cargo test --jobs 4
   ```

3. **Linting Issues**
   ```bash
   # Fix clippy warnings
   cargo clippy --fix
   
   # Format code
   cargo fmt
   
   # Check formatting
   cargo fmt --check
   ```

### Getting Help

1. **Documentation**
   - Check project documentation
   - Read code comments
   - Review examples

2. **Community**
   - Ask on Discord
   - Post on GitHub
   - Join discussions

3. **Maintainers**
   - Contact maintainers
   - Request help
   - Report issues

## Conclusion

Thank you for contributing to Hauptbuch! Your contributions help make the project better for everyone. Follow these guidelines to ensure a smooth contribution process and maintain high code quality.

For questions or concerns, please reach out to the maintainers or the community. We're here to help!
