# Contributing to Sentiment Analysis

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Development Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Sentiment-Analysis.git
cd Sentiment-Analysis

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run tests to verify setup
pytest tests/ -v
```

## Branch Naming Convention

| Type     | Prefix      | Example                     |
|----------|-------------|-----------------------------|
| Feature  | `feature/`  | `feature/add-bert-model`    |
| Bugfix   | `bugfix/`   | `bugfix/fix-negation`       |
| Hotfix   | `hotfix/`   | `hotfix/critical-patch`     |
| Docs     | `docs/`     | `docs/update-readme`        |
| Refactor | `refactor/` | `refactor/preprocessor`     |

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

**Examples:**
```
feat(preprocessing): add emoji normalization
fix(model): correct XGBoost label encoding
test(inference): add edge case tests for empty input
docs(readme): add deployment instructions
```

## Pull Request Process

1. Create a new branch from `main`
2. Make your changes
3. Run tests: `pytest tests/ -v --cov=src`
4. Run linter: `flake8 src/`
5. Commit with conventional commit messages
6. Push to your fork
7. Open a Pull Request against `main`

### PR Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] Type hints added to new functions
- [ ] Docstrings added to new functions/classes
- [ ] Tests added/updated and passing
- [ ] Code coverage maintained above 80%
- [ ] No linting errors
- [ ] README updated if applicable
- [ ] No secrets or credentials committed

## Code Style

- Follow **PEP 8** conventions
- Add **type hints** to all function signatures
- Add **docstrings** to all public functions and classes (Google style)
- Keep functions focused and small
- Use meaningful variable names

## Testing Requirements

- All new features must include tests
- Maintain minimum **80% code coverage**
- Run the full test suite before submitting PR:
  ```bash
  pytest tests/ -v --cov=src --cov-report=term-missing
  ```

## Reporting Issues

When reporting bugs, include:

1. **Description** of the bug
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (Python version, OS, package versions)

## Questions?

Open an issue on GitHub or reach out to the maintainers.
