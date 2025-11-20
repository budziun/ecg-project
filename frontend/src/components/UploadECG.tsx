import React, { useRef, useState } from 'react';
import { uploadECGFile } from '../services/api';
import ResultDisplay from './ResultDisplay';
import { TestSample, SignalData } from '../types';

interface UploadResponse {
    predicted_class: string;
    confidence: number;
    is_uncertain: boolean;
    Normal: number;
    Supraventricular: number;
    Ventricular: number;
    Fusion: number;
    Unknown: number;
    signal_raw: number[];
    signal_normalized: number[];
    threshold: number;
}

interface UploadECGProps {
    onResult?: (result: TestSample, signal: SignalData) => void;
}

export default function UploadECG({ onResult }: UploadECGProps) {
    const [isDragActive, setIsDragActive] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<TestSample | null>(null);
    const [signal, setSignal] = useState<SignalData>({ raw: [], normalized: [] });
    const inputRef = useRef<HTMLInputElement>(null);

    const validateFile = (file: File): boolean => {
        const validExtensions = ['.csv'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

        if (!validExtensions.includes(fileExtension)) {
            setError('Invalid file format. Please upload a CSV file.');
            return false;
        }

        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            setError('File too large. Maximum size is 10MB.');
            return false;
        }

        return true;
    };

    const handleFile = async (file: File) => {
        setError(null);

        if (!validateFile(file)) {
            return;
        }

        setIsLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const response: UploadResponse = await uploadECGFile(formData);

            const convertedResult: TestSample = {
                predicted_class: response.predicted_class,
                confidence: response.confidence,
                is_uncertain: response.is_uncertain,
                Normal: response.Normal,
                Supraventricular: response.Supraventricular,
                Ventricular: response.Ventricular,
                Fusion: response.Fusion,
                Unknown: response.Unknown,
                threshold: response.threshold,
                true_label: '',
                true_label_id: 0,
                index: 0,
                is_correct: false,
            };

            const signalData: SignalData = {
                raw: response.signal_raw || [],
                normalized: response.signal_normalized || []
            };

            setResult(convertedResult);
            setSignal(signalData);
            setError(null);

            if (onResult) {
                onResult(convertedResult, signalData);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to process file');
            setResult(null);
            setSignal({ raw: [], normalized: [] });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-4">
            {/* ✅ WSZYSTKO W JEDNYM KONTENERZE */}
            <div
                onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragActive(true); }}
                onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDragActive(false); }}
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setIsDragActive(false);
                    const files = e.dataTransfer?.files;
                    if (files && files[0]) handleFile(files[0]);
                }}
                onClick={() => !isLoading && inputRef.current?.click()}
                className={`border-2 border-dashed rounded-lg p-16 text-center cursor-pointer transition-all ${
                    isDragActive
                        ? 'border-blue-400 bg-blue-50 scale-[1.02]'
                        : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
                <div className="flex flex-col items-center space-y-3">
                    {/* Ikona */}
                    <svg
                        className="w-16 h-16 text-gray-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                    </svg>

                    {/* Główny tekst */}
                    <p className="text-xl font-bold text-gray-700">
                        {isLoading ? 'Processing...' : 'Drag and drop ECG file or click to select'}
                    </p>

                    {/* Supporting text */}
                    <p className="text-sm font-medium text-gray-500">
                        Supported format: CSV only
                    </p>

                    {/* Przycisk w kontenerze */}
                    {!isLoading && (
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                inputRef.current?.click();
                            }}
                            className="mt-4 px-8 py-3 bg-blue-500 text-white font-bold rounded-lg hover:bg-blue-600 transition-colors text-base shadow-md"
                        >
                            Select File
                        </button>
                    )}

                    {isLoading && (
                        <div className="flex items-center gap-2 text-blue-600">
                            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            <span className="font-semibold">Processing your file...</span>
                        </div>
                    )}
                </div>

                <input
                    ref={inputRef}
                    type="file"
                    accept=".csv"
                    onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFile(file);
                    }}
                    className="hidden"
                />
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded font-medium">
                    ⚠️ {error}
                </div>
            )}

            {result && (
                <ResultDisplay result={result} signal={signal} />
            )}
        </div>
    );
}
