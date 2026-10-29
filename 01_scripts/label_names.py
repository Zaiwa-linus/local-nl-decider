"""ラベルの ID を、結果の Markdown に書くための日本語の名前に変える。

各スクリプトから `from label_names import ja` で使う（01_scripts/ から実行すると読み込める）。
対応表にない ID は、そのまま返す。パブコメのカテゴリーは DeepSeek が設計したものなので、
11_design_categories.py を実行し直して ID が変わったときは、ここも直す。
"""

# MASSIVE のラベルの種類
MASSIVE_LABEL = {"scenario": "シナリオ", "intent": "意図"}

MASSIVE_SCENARIO = {
    "alarm": "アラーム",
    "audio": "音量",
    "calendar": "予定",
    "cooking": "料理",
    "datetime": "日時",
    "email": "メール",
    "general": "雑談・その他",
    "iot": "家電操作",
    "lists": "リスト",
    "music": "音楽の好み・設定",
    "news": "ニュース",
    "play": "再生",
    "qa": "質問",
    "recommendation": "おすすめ",
    "social": "SNS",
    "takeaway": "持ち帰り・出前",
    "transport": "交通",
    "weather": "天気",
}

MASSIVE_INTENT = {
    "alarm_query": "アラーム: 確認",
    "alarm_remove": "アラーム: 削除",
    "alarm_set": "アラーム: 設定",
    "audio_volume_down": "音量: 下げる",
    "audio_volume_mute": "音量: 消音",
    "audio_volume_other": "音量: その他",
    "audio_volume_up": "音量: 上げる",
    "calendar_query": "予定: 確認",
    "calendar_remove": "予定: 削除",
    "calendar_set": "予定: 登録",
    "cooking_query": "料理: 質問",
    "cooking_recipe": "料理: レシピ",
    "datetime_convert": "日時: 変換",
    "datetime_query": "日時: 確認",
    "email_addcontact": "メール: 連絡先の追加",
    "email_query": "メール: 確認",
    "email_querycontact": "メール: 連絡先の確認",
    "email_sendemail": "メール: 送信",
    "general_greet": "雑談: あいさつ",
    "general_joke": "雑談: 冗談",
    "general_quirky": "雑談: 変わった話",
    "iot_cleaning": "家電: 掃除機",
    "iot_coffee": "家電: コーヒー",
    "iot_hue_lightchange": "家電: 照明の色を変える",
    "iot_hue_lightdim": "家電: 照明を暗く",
    "iot_hue_lightoff": "家電: 照明を消す",
    "iot_hue_lighton": "家電: 照明をつける",
    "iot_hue_lightup": "家電: 照明を明るく",
    "iot_wemo_off": "家電: コンセントを切る",
    "iot_wemo_on": "家電: コンセントを入れる",
    "lists_createoradd": "リスト: 作成・追加",
    "lists_query": "リスト: 確認",
    "lists_remove": "リスト: 削除",
    "music_dislikeness": "音楽: 嫌い",
    "music_likeness": "音楽: 好き",
    "music_query": "音楽: 確認",
    "music_settings": "音楽: 設定",
    "news_query": "ニュース: 確認",
    "play_audiobook": "再生: オーディオブック",
    "play_game": "再生: ゲーム",
    "play_music": "再生: 音楽",
    "play_podcasts": "再生: ポッドキャスト",
    "play_radio": "再生: ラジオ",
    "qa_currency": "質問: 為替",
    "qa_definition": "質問: 言葉の意味",
    "qa_factoid": "質問: 事実",
    "qa_maths": "質問: 計算",
    "qa_stock": "質問: 株価",
    "recommendation_events": "おすすめ: イベント",
    "recommendation_locations": "おすすめ: 場所",
    "recommendation_movies": "おすすめ: 映画",
    "social_post": "SNS: 投稿",
    "social_query": "SNS: 確認",
    "takeaway_order": "持ち帰り・出前: 注文",
    "takeaway_query": "持ち帰り・出前: 確認",
    "transport_query": "交通: 確認",
    "transport_taxi": "交通: タクシー",
    "transport_ticket": "交通: 切符",
    "transport_traffic": "交通: 渋滞",
    "weather_query": "天気: 確認",
}

# パブコメの意見のカテゴリー（意見の種類）
PUBCOM_TYPE = {
    "amendment_request": "修正の要求",
    "clarification_request": "説明の要求",
    "policy_opposition": "反対",
    "new_policy_proposal": "新しい政策の提案",
    "policy_support": "賛成",
    "editorial_correction": "誤字・表記の指摘",
    "process_request": "手続きへの要望",
    "evidence_sharing": "情報提供",
    "data_request": "根拠データの要求",
    "other": "その他",
}

# パブコメの意見のカテゴリー（話題）
PUBCOM_TOPIC = {
    "public_admin_justice": "行政運営・司法",
    "digital_telecom_internet": "デジタル・通信",
    "social_health_labor": "社会保障・医療・労働",
    "finance_governance_postal": "金融・企業統治・郵政",
    "energy_nuclear": "エネルギー・原子力",
    "competition_ip_trade": "競争・取引・知財",
    "transport_infrastructure": "交通・インフラ",
    "environment_climate": "環境・気候",
    "agriculture_forestry_fisheries_animal": "農林水産・動物",
    "disaster_resilience": "防災",
    "national_security_foreign": "安全保障・外交",
    "other": "その他",
}

# livedoor ニュースのカテゴリー（連載名）
LIVEDOOR = {
    "topic-news": "一般ニュース",
    "sports-watch": "スポーツ",
    "it-life-hack": "IT",
    "kaden-channel": "家電",
    "movie-enter": "映画",
    "dokujo-tsushin": "独身女性のコラム",
    "smax": "スマートフォン",
    "livedoor-homme": "男性向け",
    "peachy": "女性向け",
}

NAMES = {**LIVEDOOR, **MASSIVE_LABEL, **MASSIVE_SCENARIO, **MASSIVE_INTENT, **PUBCOM_TYPE, **PUBCOM_TOPIC}


def ja(label):
    return NAMES.get(label, label)
