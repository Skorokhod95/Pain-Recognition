# Pain-Recognition
Программа реализует снижение размерности задачи распознавания уровня боли.
Снижение размерности реализовано отбором переменных бинарным методом роя частиц, реализованным на языке программирования общего 
назначения С++.
В качестве алгоритма классификации использована рекуррентная нейронная сеть, реализованная на базе открытой нейросетевой 
библиотеки Keras написанная на языке Python.
Взаимодействие между двумя алгоритмами происходит при помощи Python/C API [1].

[1] https://docs.python.org/3/c-api/index.html