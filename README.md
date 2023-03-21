Для запуска кода рекомендуется воспользоваться контейнером от NVIDIA https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch версии 22.11

Для установки выполните команду  docker run --gpus all -it -v <Путь на вашей машине>:/workspace -p 8888:8888 nvcr.io/nvidia/pytorch:22.11-py3
Замените <Путь на вашей машине> на путь к папке с репозиторием на локальной машине

Запустите внутри конейнера jupyter lab или jupyter notebook. Можно приатачиться с помощью Visual Studio Code с установлеными плагинами для Remote development
В браузере зайдите на http://localhost:8888 и запустится jupyter

Так же есть requirements.txt из которого можно создать свой env

Есть 3 осовных файла с выполненой работой, представляющие из себя прото-MLпайплайн. Запускаем по-очереди.
1_data_prep - ноутбук с подготовкой данных
2_modeling - ноутбук с обучением модели
3_evaluation - ноутбук с метриками качества.


Так же есть пайплайн обучения модели с пародами собак в папке dogs_breed_pipeline.
Что бы не тратить время на запус и обучение моделей, можно скачать обученные мной модели с репозитория https://huggingface.co/ulbashevsham/hometask_cats_and_dogs/tree/main

Дата сет для задания https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection?resource=download
Датасет для определения пород собак https://www.kaggle.com/competitions/dog-breed-identification

датасеты распаковываются в папку data в корне проекта. Модели так же сохраняются в папку data. Можно в другую папку, но тогда нужно будет указать в ноутбуках путь к ним. 