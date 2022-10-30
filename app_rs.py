import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger

app = FastAPI()


def batch_load_sql(query: str):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunck_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunck_dataframe)
        logger.info(f'Got chunk: {len(chunck_dataframe)}')
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def get_model_path(path: str) -> str:
    """
    Correct path for loading ML model
    """
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_features():
    """
    Uniq post_id, user_id, where were likes
    Рекомендуем те посты, которые пользователь еще не лайкал
    """
    logger.info('loading liked posts')
    liked_posts_q = """
    select 
        distinct post_id
        , user_id
    from public.feed_data
    where 1=1 
    and action = 'like'
    """
    liked_posts = batch_load_sql(liked_posts_q)

    # Фичи по постам на основе Tf-Idf

    logger.info('loading post features')
    post_features = pd.read_sql("""select * from public.posts_info_features""",
                                con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
                                    "postgres.lab.karpov.courses:6432/startml")

    # Фичи по юзерам
    logger.info('loading user features')
    user_features = pd.read_sql("""select * from public.user_data""",
                                con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
                                    "postgres.lab.karpov.courses:6432/startml")
    return [liked_posts, post_features, user_features]


def load_models():
    """
     Загрузка Catboost
    """
    model_path = get_model_path("/2_MACHINE_LEARNING/catboost_model")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model

# При поднятии сервиса положим модель и фичи в перемененные model и features соответственно
logger.info('loading model')
model = load_models()
logger.info('loading features')
features = load_features()
logger.info('service is up and running')


def get_recommended_feed(id: int, time: datetime, limit: int):
    # Загружаем фичи по пользователям
    logger.info(f'user_id: {id}')
    logger.info('reading features')
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    # Загружаем фичи по постам
    logger.info('dropping columns')
    post_features = features[1].drop(['index', 'text'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    # Объединим эти фичи
    logger.info('zipping everything')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info('assigning everything')
    user_posts_features = post_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # Добавим информацию о дате и времени рекомендации
    logger.info('add time info')
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    # Сформируем предсказания вероятности лайкнуть пост для всех постов
    logger.info('predicting')
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Убираем записи, где пользователь ранее уже ставил лайк
    logger.info('deleting previous liked posts')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts['user_id'] == id]['post_id'].values
    filtered = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Рекомендуем топ-5 постов по вероятности лайка
    recommended_posts = filtered.sort_values('predicts')[-limit:].index
    return [
        PostGet(**{
            'id': i,
            'text': content[content['post_id'] == i]['text'].values[0],
            'topic': content[content['post_id'] == i]['topic'].values[0]
        }) for i in recommended_posts
    ]


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int,
                      time: datetime,
                      limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)


