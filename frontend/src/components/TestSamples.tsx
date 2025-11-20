import React, { useState } from 'react';
import { TestSample, SignalData } from '../types';
import { api } from '../services/api';

interface TestSamplesProps {
    onSelect: (result: TestSample, signal: SignalData) => void;
}

const TestSamples: React.FC<TestSamplesProps> = ({ onSelect }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleLoadSamples = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.getTestSamples(1);
            const sample = data.samples[0];

            const signalData: SignalData = {
                raw: sample.signal_raw || [],
                normalized: sample.signal_normalized || []
            };

            onSelect(sample, signalData);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load test data');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-4">
            <button
                onClick={handleLoadSamples}
                disabled={loading}
                className="w-full px-6 py-3 bg-blue-500 text-white font-bold rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors text-base"
            >
                {loading ? 'Loading...' : 'Load one Test Sample'}
            </button>
            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded font-medium">
                    {error}
                </div>
            )}
        </div>
    );
};

export default TestSamples;
