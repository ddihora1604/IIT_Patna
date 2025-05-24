# Exam Seating Arrangement Generator

This program generates seating arrangements for exams based on timetable, classroom capacity, and student enrollment data.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following input files in the correct format:
- `input_data_tt.xlsx`: Exam timetable with columns [Date, Time, Subject]
- `input_data_classroom.xlsx`: Classroom information with columns [Building, Room, Capacity]
- `in_roll_name_mapping.xlsx`: Student roll number to name mapping with columns [Roll, Name]
- `input_data_students/{subject_code}.xlsx`: For each subject, a file containing enrolled students with column [Roll]

## Usage

Run the program using:
```bash
python seating_arrangement.py <buffer> <mode>
```

Where:
- `buffer`: Number of seats to keep empty in each room (integer)
- `mode`: Either 'sparse' or 'dense'
  - sparse: Uses 50% of effective capacity per subject
  - dense: Uses 100% of effective capacity per subject

Example:
```bash
python seating_arrangement.py 5 sparse
```

## Output

The program generates:
1. Date-wise folders in the `output` directory containing seating arrangements for morning/evening slots
2. `op_overall_seating_arrangement.xlsx`: Complete seating arrangement across all dates
3. `op_seats_left.xlsx`: Report showing remaining seats in each room
4. `errors.txt`: Log file containing any errors or clash notifications

## Error Handling

- All errors and warnings are logged to `errors.txt`
- Student roll number clashes are detected and reported
- If there are insufficient seats, appropriate error messages are displayed 