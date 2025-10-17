# Contributing to Hauptbuch Scripts

Thank you for your interest in contributing to Hauptbuch Scripts! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Script Guidelines](#script-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Getting Started

### Prerequisites

- Linux operating system
- Rust toolchain
- Docker and Docker Compose
- Git
- Basic shell scripting knowledge

### Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/hauptbuch.git
   cd hauptbuch
   ```

3. Set up the development environment:
   ```bash
   ./scripts/setup.sh
   ```

4. Run tests to ensure everything works:
   ```bash
   ./scripts/test.sh all
   ```

## Development Process

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: Feature branches
- **bugfix/**: Bug fix branches
- **hotfix/**: Critical fixes

### Workflow

1. Create a feature branch from `develop`
2. Make your changes
3. Test your changes
4. Update documentation
5. Submit a pull request

## Script Guidelines

### Naming Convention

- Use lowercase with hyphens: `script-name.sh`
- Be descriptive and clear
- Follow existing patterns

### Structure

```bash
#!/bin/bash

# Script Name
# Brief description of what the script does

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
SCRIPT_TYPE=${1:-all}
PARAMETER=${2:-default}

# Main function
main() {
    print_status "Starting script..."
    # Script logic here
    print_success "Script completed!"
}

# Handle command line arguments
case "${1:-help}" in
    "command")
        main "$@"
        ;;
    "help"|"--help"|"-h")
        echo "Script Help"
        echo "Usage: $0 [COMMAND] [PARAMETER]"
        ;;
    *)
        print_error "Unknown command: $1"
        exit 1
        ;;
esac
```

### Best Practices

1. **Error Handling**
   - Use `set -e` for strict error handling
   - Validate inputs
   - Provide meaningful error messages

2. **Output Formatting**
   - Use consistent color coding
   - Provide status updates
   - Include progress indicators

3. **Documentation**
   - Include comprehensive help text
   - Document all parameters
   - Provide usage examples

4. **Security**
   - Validate all inputs
   - Use secure defaults
   - Avoid hardcoded credentials

5. **Performance**
   - Optimize for speed
   - Use efficient algorithms
   - Minimize resource usage

## Testing

### Test Categories

1. **Unit Tests**
   - Test individual functions
   - Mock dependencies
   - Verify edge cases

2. **Integration Tests**
   - Test script interactions
   - Verify end-to-end workflows
   - Test with real dependencies

3. **Performance Tests**
   - Measure execution time
   - Test resource usage
   - Benchmark improvements

4. **Security Tests**
   - Test input validation
   - Verify access controls
   - Check for vulnerabilities

### Running Tests

```bash
# Run all tests
./scripts/test.sh all

# Run specific test types
./scripts/test.sh unit
./scripts/test.sh integration
./scripts/test.sh performance
./scripts/test.sh security

# Run with coverage
./scripts/test.sh all true
```

### Writing Tests

1. Create test files in `tests/` directory
2. Use descriptive test names
3. Test both success and failure cases
4. Include setup and teardown
5. Mock external dependencies

## Documentation

### Required Documentation

1. **Script Documentation**
   - Purpose and functionality
   - Parameters and options
   - Usage examples
   - Error handling

2. **README Updates**
   - New script entries
   - Usage examples
   - Configuration changes

3. **Changelog Updates**
   - New features
   - Bug fixes
   - Breaking changes

### Documentation Standards

- Use clear, concise language
- Include code examples
- Provide step-by-step instructions
- Keep documentation up-to-date

## Submitting Changes

### Pull Request Process

1. **Create Pull Request**
   - Use descriptive title
   - Reference related issues
   - Include detailed description

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

3. **Review Requirements**
   - All tests must pass
   - Code review approval
   - Documentation updated
   - No breaking changes without discussion

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Examples:
```
feat(scripts): add new monitoring script
fix(backup): resolve backup failure issue
docs(readme): update installation instructions
```

## Review Process

### Review Criteria

1. **Functionality**
   - Script works as intended
   - Handles edge cases
   - Provides useful output

2. **Code Quality**
   - Follows style guidelines
   - Uses best practices
   - Includes error handling

3. **Documentation**
   - Help text is comprehensive
   - Examples are accurate
   - README is updated

4. **Testing**
   - Tests cover functionality
   - Edge cases are tested
   - Performance is acceptable

### Review Timeline

- Initial review: 2-3 business days
- Follow-up reviews: 1-2 business days
- Final approval: 1 business day

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR`: Breaking changes
- `MINOR`: New features
- `PATCH`: Bug fixes

### Release Steps

1. Update version numbers
2. Update changelog
3. Create release branch
4. Run full test suite
5. Create release notes
6. Tag release
7. Deploy to production

## Getting Help

### Resources

- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/hauptbuch/hauptbuch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hauptbuch/hauptbuch/discussions)
- **Discord**: [Hauptbuch Discord](https://discord.gg/hauptbuch)

### Contact

- **Email**: team@hauptbuch.org
- **Twitter**: @HauptbuchOrg
- **GitHub**: @hauptbuch

## Recognition

Contributors will be recognized in:
- AUTHORS file
- Release notes
- Project documentation
- Community announcements

Thank you for contributing to Hauptbuch Scripts!