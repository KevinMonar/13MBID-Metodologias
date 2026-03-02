import pandas as pd


def test_great_expectations():
    """
    Test para verificar que los datos cumplen con expectativas tipo
    Great Expectations básicas.
    """

    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {
            "success_count": 0,
            "total_count": 0,
        }
    }

    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1

        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })

    add_expectation(
        "age_range",
        df["age"].between(18, 100).all(),
        "La edad debe estar entre 18 y 100 años."
    )

    add_expectation(
        "target_values",
        df["y"].isin(["yes", "no"]).all(),
        "Los valores de la columna 'y' deben ser 'yes' o 'no'."
    )

    add_expectation(
        "no_null_values",
        df.isnull().sum().sum() == 0,
        "El dataset contiene valores nulos."
    )

    add_expectation(
        "positive_duration",
        (df["duration"] >= 0).all(),
        "La duración de la llamada no puede ser negativa."
    )

    add_expectation(
        "valid_day_of_week",
        df["day_of_week"].isin(["mon", "tue", "wed", "thu", "fri"]).all(),
        "El día de la semana contiene valores inválidos."
    )

    add_expectation(
        "positive_campaign",
        (df["campaign"] > 0).all(),
        "El número de campañas debe ser mayor a 0."
    )