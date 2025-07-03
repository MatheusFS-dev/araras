from araras.keras.utils.punish import compute_params_penalized_loss
from tests.helpers import make_model


def main():
    print("Testing compute_params_penalized_loss from araras.keras.utils.punish")
    model = make_model()
    try:
        res = compute_params_penalized_loss(0.5, model, 1e-8, "add")
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
