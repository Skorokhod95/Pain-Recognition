# Pain-Recognition
Программа реализует снижение размерности задачи распознавания уровня боли.
Снижение размерности реализовано отбором переменных бинарным методом роя частиц, реализованным на языке программирования общего 
назначения С++.
В качестве алгоритма классификации использована рекуррентная нейронная сеть, реализованная на базе открытой нейросетевой 
библиотеки Keras написанная на языке Python.
Взаимодействие между двумя алгоритмами происходит при помощи Python/C API [1].

Статьи автора:
1.	Mamontov, D., Polonskaia, I., Skorokhod, A., Semenkin, E., Kessler, V., & Schwenker, F., «Evolutionary Algorithms for the Design of Neural Network Classifiers for the Classification of Pain Intensity» //IAPR Workshop on Multimodal Pattern Recognition of Social Signals in Human-Computer Interaction. – Springer, Cham, 2018. – С. 84-100
2. Скороход А. В. Нейросетевое проектирование методами эволюционных алгоритмов // материалы XXII Междунар. науч.-практ. конф., посвящ. памяти генерального конструктора ракетно-космических систем академика М. Ф. Решетнева (12–16 нояб. 2018, г. Красноярск) : в 2 ч. / под общ. ред. Ю. Ю. Логинова. Красноярск: СибГУ им. М. Ф. Решетнева, 2018. С. 163-165.

[1] https://docs.python.org/3/c-api/index.html
