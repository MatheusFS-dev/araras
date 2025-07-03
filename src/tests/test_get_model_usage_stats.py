from araras.tensorflow.utils.model import get_model_usage_stats
from tests.helpers import make_model


def main():
    print("Testing get_model_usage_stats from araras.tensorflow.utils.model")
    model = make_model()
    try:
        res = get_model_usage_stats(model, 1, "cpu", "/tmp/energy_uj", False)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
