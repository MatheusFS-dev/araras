from araras.keras.utils.summary import capture_model_summary
from tests.helpers import make_model


def main():
    print("Testing capture_model_summary from araras.keras.utils.summary")
    model = make_model()
    try:
        res = capture_model_summary(model)
        print("Result:\n", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
