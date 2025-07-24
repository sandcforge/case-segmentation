# File Insight Analysis Command

Analyze the complete logic, flow, and structure of the specified file. Provide a comprehensive breakdown including:

## 1. High-Level Overview
- Purpose and main responsibility
- Key algorithms/patterns used
- Entry points and main functions
- Design philosophy and architectural approach

## 2. Code Structure
- Class hierarchy and relationships
- Method organization and grouping
- Data structures and key variables
- Module organization patterns

## 3. Control Flow
- Main execution paths
- Decision points and branching logic
- Loop structures and iteration patterns
- State management and transitions
- Async/sync patterns

## 4. Dependencies
- Imports and external libraries
- Internal module relationships
- Configuration dependencies
- Runtime dependencies

## 5. Key Methods/Functions
- Core business logic methods
- Helper/utility functions
- Public API vs internal methods
- Critical path functions
- Entry and exit points

## 6. Data Flow
- Input sources and formats
- Data transformation processes
- Output generation and formats
- Data validation and sanitization
- Caching and persistence patterns

## 7. Error Handling
- Exception patterns and hierarchy
- Fallback mechanisms
- Validation logic
- Recovery strategies
- Logging and monitoring

## 8. Performance Considerations
- Potential bottlenecks
- Resource usage patterns
- Optimization opportunities
- Scalability concerns
- Memory management

## Usage Examples

### For a parser file:
```
/insight case_parser_channel.py

Focus on:
- The 3-step processing pipeline
- LLM integration and retry mechanisms
- Token management and truncation
- Case extraction and validation
```

### For a configuration file:
```
/insight config_manager.py

Focus on:
- Configuration loading strategies
- Environment variable handling
- Default value management
- Validation and error handling
```

### For a data processing file:
```
/insight data_preprocessor.py

Focus on:
- Data transformation pipeline
- Filtering and cleaning logic
- Output format generation
- Performance optimization
```

## Output Instructions

Save this comprehensive analysis to CLAUDE.md under a new section titled "File Analysis: [filename]" with timestamp for future development reference.

Include actionable insights and recommendations for:
- Code improvements
- Performance optimizations
- Maintainability enhancements
- Extension points for new features