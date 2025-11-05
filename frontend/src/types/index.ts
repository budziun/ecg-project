export interface ECGPrediction {
    Normal: number;
    Supraventricular: number;
    Ventricular: number;
    Fusion: number;
    Unknown: number;
    predicted_class: string;
    confidence: number;
    is_uncertain: boolean;
    threshold: number;
}

export interface TestSample extends ECGPrediction {
    true_label: string;
    true_label_id: number;
    index: number;
    is_correct: boolean;
}

export interface TestSamplesResponse {
    samples: TestSample[];
    count: number;
}

export interface BatchPredictionResponse {
    predictions: TestSample[];
    count: number;
    accuracy?: number;
    correct?: number;
}
