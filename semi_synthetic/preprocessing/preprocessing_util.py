import pandas as pd

def calculate_age(chart_date, date_of_birth):
    """
    >>> calculate_age(['2024-04-04', '2024-04-04', '2024-04-04', '2024-04-04'], ['2001-02-05 00:00:00', '2005-04-02 00:00:00', '2002-04-07 00:00:00', '2004-04-04 00:00:00'])
    [23, 19, 21, 20]
    """
    # input two arrays, first is in the format 'yyyy-mm-dd' representing the chart date and second is in the format
    # 'yyyy-mm-dd 00:00:00' representing the date of birth
    # output an array representing the ages of each patient
    if len(chart_date) != len(date_of_birth):
        return 'arrays of different length'
    
    ages = []
    for i in range(len(chart_date)):
        years = int(chart_date[i][:4]) - int(date_of_birth[i][:4])
        months = int(chart_date[i][5:7]) - int(date_of_birth[i][5:7])
        days = int(chart_date[i][8:10]) - int(date_of_birth[i][8:10])

        if months < 0:
            years -= 1
        elif months == 0 and days < 0:
            years -= 1
        
        ages.append(years)

    return ages

def truncate_text(filename, note_category):
    """
    Flan has a cap of 512 tokens for text data that we can pass through it. The prompt we use has roughly 40 tokens 
    so we need to truncate the data until it has only 470 tokens. To truncate the number of tokens, we just use the 
    split method and put the tokens back together by adding a simple whitespace.
    """
    df = pd.read_csv(filename)

    for index, row in df.iterrows():
        text = row[note_category].split()
        truncated = text[:470]
        new_text = ' '.join(truncated)

        df.at[index, note_category] = new_text

    return df

if __name__ == "__main__":
    import doctest
    doctest.testmod()