import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Set, Optional
import shutil
import time
import xlsxwriter
import logging
from datetime import datetime
import sys
from dataclasses import dataclass

# Constants
INPUT_FILE = 'input_data_tt.xlsx'
OUTPUT_DIR = Path('output')
VALID_ALLOCATION_TYPES = {'sparse', 'dense'}

@dataclass
class ExamSession:
    date: str
    time_slot: str
    subjects: List[str]

class ExamAllocation:
    # 1. Initialize and Setup Methods
    def __init__(self, spacing: int, allocation_type: str):
        self.spacing = self._validate_spacing(spacing)
        self.allocation_type = self._validate_allocation_type(allocation_type)
        self.input_file = INPUT_FILE
        
        self._validate_input_file()
        self._setup_logging()
        self._load_data()
        self._setup_output_directory()

    def _validate_spacing(self, spacing: int) -> int:
        if spacing < 0:
            raise ValueError("Spacing must be non-negative")
        return spacing

    def _validate_allocation_type(self, allocation_type: str) -> str:
        allocation_type = allocation_type.lower()
        if allocation_type not in VALID_ALLOCATION_TYPES:
            raise ValueError(f"Allocation type must be one of: {VALID_ALLOCATION_TYPES}")
        return allocation_type

    def _validate_input_file(self) -> None:
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file {self.input_file} not found")

    def _setup_logging(self) -> None:
        logging.basicConfig(
            filename='exam_allocation.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_data(self) -> None:
        print(f"Loading data from {self.input_file}...")
        self.venues = self._fetch_venue_data()
        self.student_details = self._fetch_student_mapping()
        self.exam_schedule = self._fetch_schedule()
        self.student_course_data = self._fetch_enrollment_data()

    def _setup_output_directory(self) -> None:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. Data Loading Methods
    def _fetch_venue_data(self) -> pd.DataFrame:  # Changed from _load_classroom_data
        try:
            print("Loading venue data...")
            df = pd.read_excel(self.input_file, sheet_name='in_room_capacity')
            # Rename columns to match the rest of the code
            df = df.rename(columns={
                'Room No.': 'Room',
                'Exam Capacity': 'Capacity',
                'Block': 'Building'
            })
            print(f"Loaded {len(df)} venues")
            return df
        except Exception as e:
            logging.error(f"Error loading venue data: {str(e)}")
            print(f"Error loading venue data: {str(e)}")
            raise

    def _fetch_student_mapping(self) -> Dict[str, str]:  # Changed from _load_roll_name_mapping
        try:
            print("Loading student mapping...")
            df = pd.read_excel(self.input_file, sheet_name='in_roll_name_mapping')
            # Try different possible column names
            roll_col = next((col for col in ['Roll', 'roll', 'rollno'] if col in df.columns), None)
            name_col = next((col for col in ['Name', 'name'] if col in df.columns), None)
            
            if roll_col is None or name_col is None:
                logging.warning("Could not find Roll/Name columns in student mapping")
                print("Warning: Could not find Roll/Name columns in student mapping")
                return {}
                
            mapping = dict(zip(df[roll_col].astype(str), df[name_col]))
            print(f"Loaded {len(mapping)} roll-name mappings")
            return mapping
        except Exception as e:
            logging.error(f"Error loading student mapping: {str(e)}")
            print(f"Error loading student mapping: {str(e)}")
            return {}

    def _fetch_schedule(self) -> pd.DataFrame:  # Changed from _load_timetable
        try:
            print("Loading exam schedule...")
            df = pd.read_excel(self.input_file, sheet_name='in_timetable')
            # Convert Date column to datetime with explicit format
            df['Date'] = df['Date'].astype(str)
            try:
                # Try common date formats
                df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%Y')
            except ValueError:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                except ValueError:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
                    except ValueError:
                        logging.warning("Could not parse dates with common formats, falling back to automatic parsing")
                        print("Warning: Could not parse dates with common formats, falling back to automatic parsing")
                        df['Date'] = pd.to_datetime(df['Date'])
            
            # Create separate rows for Morning and Evening sessions
            morning_df = df[['Date', 'Day', 'Morning']].copy()
            morning_df['Time'] = 'Morning'
            morning_df = morning_df.rename(columns={'Morning': 'Subject'})
            
            evening_df = df[['Date', 'Day', 'Evening']].copy()
            evening_df['Time'] = 'Evening'
            evening_df = evening_df.rename(columns={'Evening': 'Subject'})
            
            # Combine morning and evening sessions
            combined_df = pd.concat([morning_df, evening_df], ignore_index=True)
            # Remove rows where Subject is NaN
            combined_df = combined_df.dropna(subset=['Subject'])
            print(f"Loaded exam schedule with {len(combined_df)} sessions")
            return combined_df
        except Exception as e:
            logging.error(f"Error loading exam schedule: {str(e)}")
            print(f"Error loading exam schedule: {str(e)}")
            raise

    def _fetch_enrollment_data(self) -> pd.DataFrame:  # Changed from _load_course_roll_mapping
        try:
            print("Loading student course enrollment data...")
            df = pd.read_excel(self.input_file, sheet_name='in_course_roll_mapping')
            # Ensure rollno is string type
            df['rollno'] = df['rollno'].astype(str)
            print(f"Loaded {len(df)} student course enrollments")
            return df
        except Exception as e:
            logging.error(f"Error loading student course enrollment data: {str(e)}")
            print(f"Error loading student course enrollment data: {str(e)}")
            raise

    # 3. Core Processing Methods
    def _calculate_venue_capacity(self, venue_capacity: int) -> int:  # Changed from _get_effective_capacity
        effective_capacity = venue_capacity - self.spacing
        if self.allocation_type == 'sparse':
            effective_capacity = effective_capacity // 2
        return max(0, effective_capacity)

    def _generate_allocation_plan(self, date: str, time_slot: str, subjects_str: str) -> pd.DataFrame:  # Changed from _create_seating_plan
        try:
            print(f"Creating optimized exam allocation plan for {date} - {time_slot}")
            # Strip whitespace from input subjects
            subjects = [s.strip() for s in subjects_str.split(';') if s.strip()]
            
            # Calculate total students and total capacity
            total_students = 0
            subject_sizes = []
            for subject in subjects:
                student_count = len(self.student_course_data[self.student_course_data['course_code'] == subject])
                total_students += student_count
                subject_sizes.append((subject, student_count))
            
            # Calculate total available capacity
            total_capacity = sum(self._calculate_venue_capacity(venue['Capacity']) for _, venue in self.venues.iterrows())
            
            # Check if we have enough capacity
            if total_students > total_capacity:
                error_msg = f"Cannot allocate due to excess students. Total students: {total_students}, Total capacity: {total_capacity}"
                logging.error(error_msg)
                print(error_msg)
                return pd.DataFrame()  # Return empty DataFrame to indicate failure
            
            # Sort subjects by size (largest first)
            subject_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Sort rooms by capacity (largest first)
            available_venues = self.venues.sort_values(['Building', 'Capacity'], ascending=[True, False]).copy()
            allocation_plan = []
            used_venues = set()
            
            for subject, student_count in subject_sizes:
                if student_count == 0:
                    continue
                    
                # Strip whitespace from roll numbers
                students_df = self.student_course_data[self.student_course_data['course_code'] == subject]
                students = [str(roll).strip() for roll in students_df['rollno'].tolist()]
                
                # Try to keep subject in one building if possible
                buildings = available_venues['Building'].unique()
                best_building = None
                min_venues_needed = float('inf')
                
                for building in buildings:
                    venues_in_building = available_venues[available_venues['Building'] == building]
                    total_capacity = sum(self._calculate_venue_capacity(v) for v in venues_in_building['Capacity'])
                    if total_capacity >= student_count:
                        # How many venues needed?
                        cap_list = venues_in_building['Capacity'].apply(self._calculate_venue_capacity).tolist()
                        cap_list.sort(reverse=True)
                        count, acc = 0, 0
                        for cap in cap_list:
                            acc += cap
                            count += 1
                            if acc >= student_count:
                                break
                        if count < min_venues_needed:
                            min_venues_needed = count
                            best_building = building
                
                if best_building is not None:
                    venues_to_use = available_venues[available_venues['Building'] == best_building]
                else:
                    venues_to_use = available_venues
                
                for idx, (_, venue) in enumerate(venues_to_use.iterrows()):
                    if not students:
                        break
                    eff_cap = self._calculate_venue_capacity(venue['Capacity'])
                    students_in_venue = students[:eff_cap]
                    students = students[eff_cap:]
                    
                    if students_in_venue:
                        # Strip whitespace from names
                        names = [self.student_details.get(str(roll), "Unknown Name").strip() for roll in students_in_venue]
                        allocation_plan.append({
                            'Date': date,
                            'Time': time_slot,
                            'Building': venue['Building'],
                            'Room': venue['Room'],
                            'Subject': subject,
                            'Roll Numbers': ';'.join(students_in_venue),
                            'Names': ';'.join(names)
                        })
                        used_venues.add(venue['Room'])
                    
                    # Remove used venue from available_venues
                    available_venues = available_venues[available_venues['Room'] != venue['Room']]
                
                if students:
                    msg = f"Cannot allocate all students for subject {subject}. {len(students)} students remaining."
                    logging.error(msg)
                    print(msg)
            
            result_df = pd.DataFrame(allocation_plan)
            print(f"Created optimized exam allocation plan with {len(result_df)} room allocations")
            return result_df
            
        except Exception as e:
            logging.error(f"Error creating optimized exam allocation plan: {str(e)}")
            print(f"Error creating optimized exam allocation plan: {str(e)}")
            raise

    # 4. Report Generation Methods
    def _generate_vacancy_report(self, final_plan: pd.DataFrame):  # Changed from _generate_seats_left_report
        try:
            # Aggregate number of unique subjects per room across all slots
            if not final_plan.empty and 'Room' in final_plan.columns and 'Subject' in final_plan.columns:
                room_alloted = final_plan.groupby('Room')['Subject'].nunique().reset_index().rename(columns={'Subject': 'Alloted'})
            else:
                room_alloted = pd.DataFrame(columns=['Room', 'Alloted'])

            # Merge with room master list
            vacancy_df = self.venues.copy()
            vacancy_df = vacancy_df.rename(columns={'Room': 'Room No.', 'Capacity': 'Exam Capacity', 'Building': 'Block'})
            vacancy_df = vacancy_df.merge(room_alloted, left_on='Room No.', right_on='Room', how='left')
            vacancy_df['Alloted'] = vacancy_df['Alloted'].fillna(0).astype(int)
            vacancy_df['Vacant (B-C)'] = vacancy_df['Exam Capacity'] - vacancy_df['Alloted']

            # Only keep required columns
            output_columns = [
                'Room No.',
                'Exam Capacity',
                'Block',
                'Alloted',
                'Vacant (B-C)'
            ]
            vacancy_df = vacancy_df[output_columns]
            vacancy_file = os.path.join(OUTPUT_DIR, 'op_seats_left.xlsx')
            print(f"Saving vacancy report to: {vacancy_file}")
            try:
                vacancy_df.to_excel(vacancy_file, index=False, engine='openpyxl')
                print(f"Successfully saved vacancy report")
            except Exception as e:
                error_msg = f"Failed to save vacancy report: {str(e)}"
                logging.error(error_msg)
                print(error_msg)
                raise
        except Exception as e:
            logging.error(f"Error generating vacancy report: {str(e)}")
            print(f"Error generating vacancy report: {str(e)}")
            raise

    # 5. Main Process Method
    def create_exam_allocation(self):  # Changed from generate_seating_arrangement
        try:
            if self.exam_schedule.empty:
                logging.error("No exam schedule data available")
                print("Error: No exam schedule data available")
                return
                
            # Clear and create output directory
            print(f"Cleaning up output directory: {OUTPUT_DIR}")
            try:
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                    print(f"Removed existing output directory: {OUTPUT_DIR}")
                time.sleep(1)  # Give the system time to clean up
                os.makedirs(OUTPUT_DIR)
                print(f"Created output directory: {OUTPUT_DIR}")
            except Exception as e:
                logging.error(f"Error cleaning up output directory: {str(e)}")
                print(f"Error cleaning up output directory: {str(e)}")
                # Try to proceed anyway
                os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            all_allocation_plans = []
            processed_slots = set()  # Track processed date-time slots
            
            # Group by Date and Time
            # --- Get all unique dates and always create both morning and evening folders ---
            unique_dates = self.exam_schedule['Date'].dt.strftime('%d_%m_%Y').unique()
            for date_str in unique_dates:
                date_dir = os.path.join(OUTPUT_DIR, date_str)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                    print(f"Created date directory: {date_dir}")
                for time_slot in ['Morning', 'Evening']:
                    time_dir = os.path.join(date_dir, time_slot.lower())
                    if not os.path.exists(time_dir):
                        os.makedirs(time_dir)
                        print(f"Created time slot directory: {time_dir}")

            # Now process each slot as before
            for _, row in self.exam_schedule.iterrows():
                try:
                    date = row['Date']
                    time_slot = row['Time']
                    subject_str = row['Subject']

                    # Robust NO EXAM check
                    if pd.isna(subject_str) or subject_str.strip().upper() == "NO EXAM":
                        print(f"Skipping {date.strftime('%d_%m_%Y')} - {time_slot}: No exam")
                        continue

                    # Skip if we've already processed this date-time slot
                    slot_key = (date.strftime('%d_%m_%Y'), time_slot)
                    if slot_key in processed_slots:
                        continue
                    processed_slots.add(slot_key)

                    # Create date directory (already created above)
                    date_str = date.strftime('%d_%m_%Y')
                    date_dir = os.path.join(OUTPUT_DIR, date_str)

                    # Create time slot directory under date directory (already created above)
                    time_dir = os.path.join(date_dir, time_slot.lower())

                    # --- Clash checking for this slot ---
                    subject_list = [s.strip() for s in subject_str.split(';') if s.strip()]
                    rolls_by_subject = {}
                    for subject in subject_list:
                        rolls = set(self.student_course_data[self.student_course_data['course_code'] == subject]['rollno'].astype(str))
                        rolls_by_subject[subject] = rolls
                    clash_found = False
                    subjects = list(rolls_by_subject.keys())
                    for i in range(len(subjects)):
                        for j in range(i + 1, len(subjects)):
                            s1, s2 = subjects[i], subjects[j]
                            intersection = rolls_by_subject[s1] & rolls_by_subject[s2]
                            if intersection:
                                clash_found = True
                                for roll in intersection:
                                    msg = f"Clash detected: {roll} in both {s1} and {s2} on {date.strftime('%d_%m_%Y')} {time_slot}"
                                    print(msg)
                                    logging.error(msg)
                    if clash_found:
                        print(f"Clash found in slot {date.strftime('%d_%m_%Y')} {time_slot}. See errors.txt for details.")

                    # Generate seating plan for this slot
                    allocation_plan = self._generate_allocation_plan(
                        date_str, 
                        time_slot, 
                        subject_str
                    )

                    if not allocation_plan.empty:
                        # Save time slot specific plan in the time slot directory
                        slot_file = os.path.join(time_dir, f'allocation_plan.xlsx')
                        try:
                            print(f"Saving allocation plan to: {slot_file}")
                            allocation_plan.to_excel(slot_file, index=False, engine='openpyxl')
                            print(f"Successfully saved allocation plan for {date_str} - {time_slot}")
                            all_allocation_plans.append(allocation_plan)

                            # --- Generate attendance sheets for each subject and room ---
                            for _, row_sp in allocation_plan.iterrows():
                                subject = row_sp['Subject']
                                room = row_sp['Room']
                                roll_numbers = row_sp['Roll Numbers'].split(';') if row_sp['Roll Numbers'] else []
                                names = row_sp['Names'].split(';') if row_sp['Names'] else []
                                # Prepare attendance DataFrame
                                attendance_data = []
                                for roll, name in zip(roll_numbers, names):
                                    attendance_data.append({
                                        'Roll': roll,
                                        'Student Name': name,
                                        'Signature': ''
                                    })
                                attendance_df = pd.DataFrame(attendance_data)
                                # Write to Excel with merged header row, then column headers, then attendance table using xlsxwriter for rich text and black/white formatting
                                attendance_file = os.path.join(time_dir, f'{date_str}_{subject}_{room}.xlsx')
                                workbook = xlsxwriter.Workbook(attendance_file)
                                worksheet = workbook.add_worksheet()
                                # Set column widths to match the image
                                worksheet.set_column('A:A', 30)
                                worksheet.set_column('B:B', 40)
                                worksheet.set_column('C:C', 30)
                                # Header formats
                                header_bg = '#222222'  # dark gray/black
                                header_format = workbook.add_format({'bold': True, 'font_color': 'white', 'align': 'center', 'valign': 'vcenter', 'bg_color': header_bg, 'font_size': 12, 'border': 1})
                                header_rich = [
                                    header_format, 'Course: ', header_format, f'{subject} | ',
                                    header_format, 'Room: ', header_format, f'{room} | ',
                                    header_format, 'Date: ', header_format, f'{date_str.replace("_", "-")} | ',
                                    header_format, 'Session: ', header_format, f'{time_slot}'
                                ]
                                worksheet.merge_range('A1:C1', '', header_format)
                                worksheet.write_rich_string('A1', *header_rich)
                                worksheet.set_row(0, 30)
                                # Column headers
                                col_header_format = workbook.add_format({'bold': True, 'font_color': 'white', 'align': 'center', 'valign': 'vcenter', 'bg_color': header_bg, 'border': 1})
                                worksheet.write('A2', 'Roll', col_header_format)
                                worksheet.write('B2', 'Student Name', col_header_format)
                                worksheet.write('C2', 'Signature', col_header_format)
                                # Data rows
                                border_format = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
                                for idx, row_data in enumerate(attendance_data, start=2):
                                    worksheet.write(idx, 0, row_data['Roll'], border_format)
                                    worksheet.write(idx, 1, row_data['Student Name'], border_format)
                                    worksheet.write(idx, 2, '', border_format)
                                # Leave one empty row
                                idx += 1
                                worksheet.write(idx, 0, '', border_format)
                                worksheet.write(idx, 1, '', border_format)
                                worksheet.write(idx, 2, '', border_format)
                                # Add TA and Invigilator lines after the empty row
                                extra_lines = ['TA1', 'TA2', 'TA3', 'TA4', 'TA5', 'Invigilator1', 'Invigilator2', 'Invigilator3', 'Invigilator4', 'Invigilator5']
                                for i, label in enumerate(extra_lines):
                                    worksheet.write(idx + 1 + i, 0, label, border_format)
                                    worksheet.write(idx + 1 + i, 1, '', border_format)
                                    worksheet.write(idx + 1 + i, 2, '', border_format)
                                workbook.close()
                        except Exception as e:
                            logging.error(f"Error saving allocation plan for {date_str} - {time_slot}: {str(e)}")
                            print(f"Error saving allocation plan for {date_str} - {time_slot}: {str(e)}")
                    else:
                        print(f"No allocation plan generated for {date_str} - {time_slot}")

                except Exception as e:
                    logging.error(f"Error processing date {date}, time {time_slot}: {str(e)}")
                    print(f"Error processing date {date}, time {time_slot}: {str(e)}")
                    continue

            # Combine all allocation plans and save overall arrangement
            if all_allocation_plans:
                final_plan = pd.concat(all_allocation_plans, ignore_index=True)

                # --- Generate vacancy report BEFORE renaming columns ---
                try:
                    print("Generating vacancy report...")
                    self._generate_vacancy_report(final_plan)
                    print("Successfully created vacancy report")
                except Exception as e:
                    logging.error(f"Error generating vacancy report: {str(e)}")
                    print(f"Error generating vacancy report: {str(e)}")

                # --- Add Day column by mapping Date to Day from timetable ---
                # Prepare a mapping from date_str to Day
                timetable_map = self.exam_schedule.copy()
                timetable_map['Date_str'] = timetable_map['Date'].dt.strftime('%d_%m_%Y')
                date_to_day = dict(zip(timetable_map['Date_str'], timetable_map['Day']))
                final_plan['Day'] = final_plan['Date'].map(date_to_day)

                # --- Add Allocated_students_count column ---
                final_plan['Allocated_students_count'] = final_plan['Roll Numbers'].apply(lambda x: len(x.split(';')) if isinstance(x, str) and x else 0)

                # --- Rename columns to match screenshot ---
                final_plan = final_plan.rename(columns={
                    'Date': 'Date',
                    'Day': 'Day',
                    'Subject': 'course_code',
                    'Room': 'Room',
                    'Allocated_students_count': 'Allocated_students_count',
                    'Roll Numbers': 'Roll_list (semicolon separated)'
                })

                # --- Reorder columns as per screenshot ---
                output_columns = [
                    'Date',
                    'Day',
                    'course_code',
                    'Room',
                    'Allocated_students_count',
                    'Roll_list (semicolon separated)'
                ]
                final_plan = final_plan[output_columns]

                # Save overall seating arrangement
                overall_file = os.path.join(OUTPUT_DIR, 'op_overall_seating_arrangement.xlsx')
                try:
                    print(f"Saving overall seating arrangement to: {overall_file}")
                    final_plan.to_excel(overall_file, index=False, engine='openpyxl')
                    print("Successfully created overall seating arrangement")
                except Exception as e:
                    logging.error(f"Error saving overall seating arrangement: {str(e)}")
                    print(f"Error saving overall seating arrangement: {str(e)}")
            else:
                error_msg = "No seating plans were generated"
                logging.error(error_msg)
                print(f"Error: {error_msg}")
                # Still generate an empty seats left report for structure
                try:
                    print("Generating empty vacancy report...")
                    self._generate_vacancy_report(pd.DataFrame())
                    print("Successfully created empty vacancy report")
                except Exception as e:
                    logging.error(f"Error generating empty vacancy report: {str(e)}")
                    print(f"Error generating empty vacancy report: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error in create_exam_allocation: {str(e)}")
            print(f"Error in create_exam_allocation: {str(e)}")
            raise

# 6. Entry Point Functions
def run_allocation():
    try:
        if len(sys.argv) != 3:
            print("Usage: python exam_allocation.py <spacing> <allocation_type>")
            print("Allocation type should be either 'sparse' or 'dense'")
            sys.exit(1)
        
        spacing = int(sys.argv[1])
        allocation_type = sys.argv[2]
        
        if allocation_type.lower() not in ['sparse', 'dense']:
            print("Allocation type should be either 'sparse' or 'dense'")
            sys.exit(1)
        
        print(f"Starting exam allocation with spacing={spacing}, type={allocation_type}")
        
        allocation = ExamAllocation(spacing, allocation_type)
        allocation.create_exam_allocation()
        
        print("Exam allocation completed successfully!")
        print("Output files are in the 'output' directory")
        
    except Exception as e:
        logging.error(f"Error in execution: {str(e)}")
        print(f"An error occurred: {str(e)}")
        print("Please check errors.txt for details.")
        sys.exit(1)

if __name__ == "__main__":
    run_allocation()