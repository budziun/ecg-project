export interface ECGPrediction {
    predicted_class: string;
    confidence: number;
    probabilities: {
        Normal: number;
        Supraventricular: number;
        Ventricular: number;
        Fusion: number;
        Unknown: number;
    };
}

export interface TestSample {
    predicted_class: string;
    confidence: number;
    is_uncertain: boolean;
    Normal: number;
    Supraventricular: number;
    Ventricular: number;
    Fusion: number;
    Unknown: number;
    threshold: number;
    true_label?: string;
    true_label_id?: number;
    index?: number;
    is_correct?: boolean;
    signal_raw?: number[];
    signal_normalized?: number[];
}

export interface TestSamplesResponse {
    samples: TestSample[];
    count: number;
}

export interface BatchPredictionResponse {
    predictions: TestSample[];
    count: number;
    accuracy: number;
    correct: number;
}

export interface SignalData {
    raw: number[];
    normalized: number[];
}
