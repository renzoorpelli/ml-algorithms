from tinygrad import Context, GlobalCounters, Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist


class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2, 2))
        x = self.l2(x).relu().max_pool2d((2, 2))
        return self.l3(x.flatten(1).dropout(0.5))


def main():

    X_train, Y_train, X_test, Y_test = mnist()
    print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)

    # create the Model
    model = Model()
    acc = (
        model(X_test).argmax(axis=1) == Y_test
    ).mean()  # ~10% accuracy, expected from random model
    print(f"Initial accuracy %{acc.item()}")

    optim = nn.optim.Adam(nn.state.get_parameters(model))
    batch_size = 128

    def step() -> Tensor:
        Tensor.training = True  # makes dropout work
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        X, Y = X_train[samples], Y_train[samples]
        optim.zero_grad()
        loss = model(X).sparse_categorical_crossentropy(Y).backward()
        optim.step()
        return loss

    # breakdown time by kernel
    # GlobalCounters.reset()
    # with Context(DEBUG=2):

    jit_step = TinyJit(step)
    # train_res = timeit.repeat(jit_step, repeat=5, number=1)
    # print(train_res)

    for stp in range(7000):
        loss = jit_step()
        if stp % 100 == 0:
            Tensor.training = False
            acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
            print(f"step {stp:4d}, loss {loss.item():.2f}, acc {acc * 100.0:.2f}%")


if __name__ == "__main__":
    main()
