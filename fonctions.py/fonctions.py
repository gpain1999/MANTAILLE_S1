def convert_duration_to_minutes(duration_str):
    """
    Convertit une chaîne de caractères représentant une durée au format "0 days 00:06:020"
    en minutes.
    
    :param duration_str: Chaîne de caractères de la durée.
    :return: Durée en minutes.
    """
    # Extraire les jours, heures, minutes et secondes
    days, time = duration_str.split(' days ')
    hours, minutes, seconds = map(int, time.split(':'))

    # Convertir en minutes
    total_minutes = int(days) * 24 * 60 + hours * 60 + minutes + seconds / 60
    return total_minutes