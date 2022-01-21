from .pipeline import MeanDummyRegressor, PredictCAO
from .feature_engineering import FeatureEngineeringCAO
from .data_examples import (
    CHARGE_CHEMISTRY,
    LIMESTONE_CONSUMPTIONS,
    CHARGE_CONSUMPTIONS,
    COKE_CONSUMPTIONS,
    COKE_SIEVING,
    LIMESTONE_CAO
)


def main():
    model = MeanDummyRegressor()
    feature_engineering_ = FeatureEngineeringCAO()

    predictor = PredictCAO(
        model=model,
        feature_engineering=feature_engineering_)

    predicted_value = predictor.predict(CHARGE_CHEMISTRY,
                                        LIMESTONE_CONSUMPTIONS,
                                        CHARGE_CONSUMPTIONS,
                                        COKE_CONSUMPTIONS,
                                        COKE_SIEVING,
                                        LIMESTONE_CAO)

    print(f'predicted_value = {predicted_value}')


main()
