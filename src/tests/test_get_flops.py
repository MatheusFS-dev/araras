from araras.keras.utils.profiler import get_flops
from tests.helpers import make_model


def main():
    print("Testing get_flops from araras.keras.utils.profiler")
    model = make_model()
    try:
        res = get_flops(model, 1)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
