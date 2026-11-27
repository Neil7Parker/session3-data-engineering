from datetime import datetime

class GradeTracker:
    def __init__(self, course_name):
        self.course_name = course_name
        self.student_grades = {}
        self.last_updated = datetime.now()

    def add_grade(self, student_name, grade):
        self.student_grades[student_name] = grade
        self.last_updated = datetime.now()

    def get_average(self):
        if not self.student_grades:
            return 0
        return sum(self.student_grades.values()) / len(self.student_grades)

    def get_top_student(self):
        if not self.student_grades:
            return None
        return max(self.student_grades.items(), key=lambda x: x[1])

    def __str__(self):
        return f"GradeTracker for '{self.course_name}' with {len(self.student_grades)} students"
