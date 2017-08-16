import csv

def import_glucose_from_csv(csv_file):
    """
    Import glucose CSV data.

    We'll process all rows first and create Glucose model objects from them
    and perform a bulk create. This way, no records will be inserted unless
    all records are good.

    Also note that we're using splitlines() to make sure 'universal newlines'
    is used.

    Assumed order: value, category, record_date, record_time, notes
    """
    csv_data = []
    reader = csv.reader(csv_file.read().splitlines(), delimiter=',',
                        quotechar='"')
    for row in reader:
        csv_data.append([item.strip() for item in row])

    glucose_objects = []

    # Check if headers exists. Skip the first entry if true.
    header_check = ['value', 'category', 'date', 'time']
    first_row = [i.lower().strip() for i in csv_data[0]]
    if all(i in first_row for i in header_check):
        csv_data = csv_data[1:]

    for row in csv_data:
        # Let's do an extra check to make sure the row is not empty.
        if row:
            try:
                category = Category.objects.get(name__iexact=row[1].strip())
            except ObjectDoesNotExist:
                category = Category.objects.get(name__iexact='No Category'.strip())

            # Since we always store the value in mg/dL format in the db, we need
            # to make sure we convert it here if the user's setting is set to
            # mmol/L.
            if user.settings.glucose_unit.name == 'mmol/L':
                value = int(to_mg(row[0]))
            else:
                value = int(row[0])
            print(value)
