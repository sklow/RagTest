"""
包括的テストケース定義
TMC4361Aデータシートに対するRAGシステム評価用
"""

TEST_CASES = [
    # === カテゴリ1: 具体的数値検索 (Factual/Numeric) ===
    {
        "query": "What is the supply voltage range for TMC4361A?",
        "category": "factual_numeric",
        "expected_keywords": ["voltage", "VCC", "3.3", "5", "supply"],
        "difficulty": "easy",
        "expected_best_method": "sparse",
        "description": "電源電圧の具体的な数値を問う"
    },
    {
        "query": "What is the maximum SPI clock frequency?",
        "category": "factual_numeric",
        "expected_keywords": ["SPI", "clock", "frequency", "MHz"],
        "difficulty": "medium",
        "expected_best_method": "hybrid",
        "description": "SPI通信の最大クロック周波数を問う"
    },
    {
        "query": "Operating temperature range",
        "category": "factual_numeric",
        "expected_keywords": ["temperature", "°C", "-40", "125", "operating"],
        "difficulty": "easy",
        "expected_best_method": "sparse",
        "description": "動作温度範囲の具体的数値を問う"
    },

    # === カテゴリ2: 概念的質問 (Conceptual) ===
    {
        "query": "How does the S-shaped ramp generator work?",
        "category": "conceptual",
        "expected_keywords": ["ramp", "velocity", "acceleration", "profile", "S-shaped"],
        "difficulty": "medium",
        "expected_best_method": "dense",
        "description": "S字ランプ生成の仕組みを問う"
    },
    {
        "query": "What is the purpose of closed-loop operation?",
        "category": "conceptual",
        "expected_keywords": ["closed-loop", "encoder", "feedback", "position", "control"],
        "difficulty": "medium",
        "expected_best_method": "dense",
        "description": "クローズドループ制御の目的を問う"
    },
    {
        "query": "How does the encoder interface function?",
        "category": "conceptual",
        "expected_keywords": ["encoder", "interface", "incremental", "position", "ABN"],
        "difficulty": "medium",
        "expected_best_method": "dense",
        "description": "エンコーダインターフェースの機能を問う"
    },

    # === カテゴリ3: 専門用語・略語 (Technical Terms) ===
    {
        "query": "XTARGET register description",
        "category": "technical_terms",
        "expected_keywords": ["XTARGET", "register", "target", "position"],
        "difficulty": "easy",
        "expected_best_method": "sparse",
        "description": "XTARGETレジスタの説明を問う"
    },
    {
        "query": "ChopSync feature explanation",
        "category": "technical_terms",
        "expected_keywords": ["ChopSync", "chopper", "synchronization"],
        "difficulty": "hard",
        "expected_best_method": "sparse",
        "description": "ChopSync機能の説明を検索"
    },
    {
        "query": "SixPoint ramp mode",
        "category": "technical_terms",
        "expected_keywords": ["SixPoint", "ramp", "motion", "profile"],
        "difficulty": "medium",
        "expected_best_method": "sparse",
        "description": "SixPointランプモードの検索"
    },

    # === カテゴリ4: 複合条件検索 (Multi-aspect) ===
    {
        "query": "SPI communication protocol for motor speed control",
        "category": "multi_aspect",
        "expected_keywords": ["SPI", "speed", "control", "register", "velocity"],
        "difficulty": "hard",
        "expected_best_method": "hybrid",
        "description": "SPI通信とモーター速度制御の複合検索"
    },
    {
        "query": "Encoder feedback for position tracking accuracy",
        "category": "multi_aspect",
        "expected_keywords": ["encoder", "position", "feedback", "accuracy", "deviation"],
        "difficulty": "hard",
        "expected_best_method": "hybrid",
        "description": "エンコーダフィードバックと位置追跡精度の複合検索"
    },
    {
        "query": "Reference switch configuration and homing procedure",
        "category": "multi_aspect",
        "expected_keywords": ["reference", "switch", "homing", "configuration"],
        "difficulty": "hard",
        "expected_best_method": "hybrid",
        "description": "リファレンススイッチ設定とホーミング手順の複合検索"
    },

    # === カテゴリ5: 意味的類似検索 (Semantic Similarity) ===
    {
        "query": "How to make the motor move smoothly without vibration",
        "category": "semantic_similarity",
        "expected_keywords": ["ramp", "velocity", "smooth", "jerk", "acceleration"],
        "difficulty": "hard",
        "expected_best_method": "dense",
        "description": "振動なくスムーズに動かす方法（直接的なキーワードなし）"
    },
    {
        "query": "Preventing the motor from losing track of its location",
        "category": "semantic_similarity",
        "expected_keywords": ["position", "encoder", "closed-loop", "stall", "deviation"],
        "difficulty": "hard",
        "expected_best_method": "dense",
        "description": "位置ロストの防止（間接的な表現）"
    },
    {
        "query": "Connecting the chip to a microcontroller",
        "category": "semantic_similarity",
        "expected_keywords": ["SPI", "interface", "communication", "microcontroller", "connection"],
        "difficulty": "medium",
        "expected_best_method": "dense",
        "description": "マイコン接続方法（chipという間接表現）"
    },
]
