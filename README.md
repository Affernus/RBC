# Цифровой прорыв
# Всероссийский чемпионат
# Задача от РБК: радар тенденций новостных статей

Репозиторий содержит код и данные для решения задачи в рамках Всероссийского чемпионата "Цифровой прорыв: искусственный интеллект"

У компании РБК довольно взрослая аудитория, которую она хочет расширить за счет добавления статей на актуальные темы. Для этого нужно проанализировать лучшие новости российских СМИ и научиться предсказывать их популярность.

Цель — предсказать 3 численные характеристики, которые в полной мере показывают популярность статьи: *views, full reads percent, depth*.

Для оценки качества решения используется метрика R2.

```R2_result = 0.4 * R2_views + 0.3 * R2_depth + 0.3 * R2_full_reads_percent```

Результат представленного кода на public leaderboard: ```R2_result=0.774``` (топ-2)

## Локальная установка

1. Создайте виртуальное окружение на python 3.9.4 (рекомендуется для точной совместимости версий). Если вы используете pyenv с плагином pyenv-virtualenv:
   ```pyenv virtualenv 3.9.4 rbc```

2. Клонируйте репозиторий или скачайте zip архив https://github.com/Affernus/RBC/archive/refs/heads/master.zip

   ```git clone git@github.com:Affernus/RBC.git```

3. Убедитесь, что находитесь в корневой директории проекта,
если нет, перейдите туда

4. Активируйте виртуальное окружение. Если вы используете pyenv с плагином pyenv-virtualenv:

   ```pyenv activate rbc```

5. Обновите pip 

   ```pip install --upgrade pip```

6. Установите requirements 

   ```pip install -r requirements.txt```

7. Установите репозиторий в виде библиотеки 

   ```pip install -e ./.```

## Структура проекта

1. ```data``` - папка с данными
   - ```data/embeddings``` - папка с эмбеддингами авторов, статей, заголовков (см. подробнее notebooks/embeddings_and_word_models.ipynb)
   - ```data/external``` - папка с внешними данными
      - ```data/external/covid``` - данные о заболеваемости и смертности от covid-19 с сайта https://ourworldindata.org
      - ```data/external/economic``` - экономические показатели с сайта https://ru.investing.com/ (индексы из таблиц копипастились вручную, по всем индексам таблицы короткие и это не представляет сложности)
      - ```data/external/parsing``` - результат парсинга сайта https://www.rbc.ru (парсер - src/rbk/utils/parser.py)
      - ```data/external/weather``` - данные о погоде в Москве с сайта https://rp5.ru/
   - ```data/prepared``` - входные данные, обогащенные признаками, рассчитанными на основе внешних данных (см. подробнее notebooks/eda_and_prepare_data.ipynb)
   - ```raw``` - исходные данные задачи
   - ```submits``` - файлы отправок решений
<!--  -->
2. ```docs``` - документы по задаче (описание, условия)
<!--  -->
3. ```notebooks``` - тетрадки проекта
   - ```notebooks/create_submission.ipynb``` - тетрадка, формирующая файл с решением. Подтягивает модельки, данные, запускает модельки на данных, получает предикты и записывает в файл в нужном формате
   - ```notebooks/eda_and_prepare_data.ipynb``` - тетрадка с исследованиями датасета, подготовкой признаков на основе внешних данных и сохранением результата в ```data/prepared```
   - ```notebooks/embeddings_and_word_models.ipynb``` - тетрадка с подготовкой текстов для обучения word2vec, обучением, расчетом эмбеддингов и сохранением полученных результатов в ```data/embeddings``` и ```src/models/word2vec```
   - ```notebooks/train_model_depth.ipynb```, ```notebooks/train_model_frp.ipynb```, ```notebooks/train_model_views.ipynb``` - тетрадки с подбором гиперпараметров и признаков, кросс-валидацией и обучением моделей для прогноза глубины просморенного материала, процента читателей, полностью прочитавших статью и количества просмотров статьи
<!--  -->
4. ```scr``` - библиотеки и модели проекта
   - ```src/models``` - содержит модели проекта
      - ```src/models/nltk/corpora/stopwords``` - стоп-слова библиотеки nltk
      - ```src/models/word2vec``` - предобученная модель word2vec (см. ```notebooks/embeddings_and_word_models.ipynb```)
      - ```src/models/model_depth.pickle, src/models/model_depth_train_cols.json``` - модель прогноза depth и названия признаков, которы используются моделью;
      - ```src/models/model_full_reads_percent.pickle, src/models/model_full_reads_percent_train_cols.json``` - модель прогноза full reads percent и названия признаков, которые используются моделью;
      - ```src/models/model_views.pickle, src/models/model_views_train_cols.json``` - модель прогноза views и названия признаков, которые используются моделью

   - ```src/rbk/utils``` - библиотека со скриптами для проекта
      - ```src/rbk/utils/global_vars.py``` - содержит глобальные константы
      - ```src/rbk/utils/embeddings.py``` - скрипты для расчета и обработки эмбеддингов и модели word2vec
      - ```src/rbk/utils/helpers.py``` - неклассифицированные вспомогательные функции
      - ```src/rbk/utils/ml_scripts.py``` - скрипты, относящиеся к построению и валидации моделей машинного обучения
      - ```src/rbk/utils/my_plotlib.py``` - скрипты, связанные с построением графиков
      - ```src/rbk/utils/parser.py``` - парсер сайта РБК
      - ```src/rbk/utils/prepare_data.py``` - скрипты для подготовки данных (предобработка, подготовка внешних данных, расчет признаков, обогащение входных данных и т.д.)
      - ```src/rbk/utils/text_processing.py``` - скрипты для обработки текста и расчета фич, связанных с readability


## Запуск и получение прогноза

Важно: при запуске тетрадок, если вы устанавливали виртуальную среду, не забудьте выбрать соответствующий kernel в среде разработки.

1. В файле ```src/rbk/utils/global_vars.py``` поменять значение константы ```ROOT_ABSPATH``` на путь к корневой директории проекта (у меня это ```'/Users/affernus/PROJECTS/hacks/RBC'```)

2. Вариант без обучения моделей заново: запустить тетрадку ```notebooks/create_submission.ipynb```. После завершения прогона тетрадки в папке ```data/submits``` появится файл с названием ```submits_{YYYYMMDDhhmmss}.csv``` Это результат прогноза.

   Важно: модельки сохранены в pickle (т.к. они кастомные), поэтому важно, чтобы была создана и настроена виртуальная среда, как описано в блоке "Локальная установка". Если модели не загружаются в связи с конфликтом версии python, можно либо выполнить установку среды согласно "Локальная установка", либо обучить их заново, процесс описан ниже.

3. Если нужно заново обучить модели прогноза views, full_reads_percent, depth: последовательно выполнить тетрадки ```notebooks/train_model_depth.ipynb, notebooks/train_model_frp.ipynb, notebooks/train_model_views.ipynb``` не меняя в них параметров (иначе модели обучатся не так, как ранее). Файлы моделей обновятся, затем нужно запустить ```notebooks/create_submission.ipynb```

4. Если нужно заново рассчитать эмбеддинги: запустить ```notebooks/embeddings_and_word_models.ipynb```. Также там можно заново обучить модель word2vec, но, поскольку она обучается многопоточно, то результат обучения будет отличаться и дальнейший прогноз моделей не будет повторяться.

5. Если нужно заново рассчитать признаки и сформировать данные в ```data/prepared```: запустить тетрадку ```notebooks/eda_and_prepare_data.ipynb```


## Полный пайплайн с расчетом всех признаков и обучением моделей заново:

Важно: при запуске тетрадок, если вы устанавливали виртуальную среду, не забудьте выбрать соответствующий kernel в среде разработки.

1. В файле ```src/rbk/utils/global_vars.py``` поменять значение константы ```ROOT_ABSPATH``` на путь к корневой директории проекта (у меня это ```'/Users/affernus/PROJECTS/hacks/RBC'```)

2. запустить тетрадку ```notebooks/eda_and_prepare_data.ipynb``` (результат выполнения - файлы train.parquet и test.parquet в ```data/prepared```)

3. запустить тетрадку ```notebooks/embeddings_and_word_models.ipynb``` (результат выполнения - файлы в ```data/embeddings```: authors_embs.parquet, first_part_embs.parquet, full_embs.parquet, last_part_embs.parquet, tags_embs.parquet, titles_emb.parquet, topn_similar.parquet)

4. последовательно выполнить тетрадки ```notebooks/train_model_depth.ipynb, notebooks/train_model_frp.ipynb, notebooks/train_model_views.ipynb``` (результат выполения - файлы в ```src/models```: model_depth_train_cols.json, model_full_reads_percent_train_cols.json, model_views_train_cols.json, model_depth.pickle, model_full_reads_percent.pickle, model_views.pickle)

5. запустить тетрадку ```notebooks/create_submission.ipynb```. После завершения прогона тетрадки в папке data/submits появится файл с названием submits_{YYYYMMDDhhmmss}.csv Это результат прогноза.