from train_test_split import train_test_split
from random import randint
from collections import Counter

X = [str(x)+'_right.jpg' for x in range(100)]
X += [str(x)+'_left.jpg' for x in range(100)]

y = [randint(1,4) for _ in range(200)]

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(Counter(y_train), Counter(y_test))
