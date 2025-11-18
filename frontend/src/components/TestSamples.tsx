import React, { useState } from 'react';
import { TestSample } from '../types';
import { api } from '../services/api';

interface TestSamplesProps {
    onSelect: (result: TestSample, signal: number[]) => void;
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

            // Generuj sygnał z probabilities (demo)
            const signal = Array(187).fill(0).map(() => Math.random() * 0.5);

            onSelect(sample, signal);
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
                className="w-full bg-green-500 hover:bg-green-600 disabled:bg-gray-400
          text-white font-bold py-2 px-4 rounded-lg transition"
            >
                {loading ? 'Loading...' : 'Load one Test Sample'}
            </button>

            {error && <p className="text-red-600">{error}</p>}
        </div>
    );
};

export default TestSamples;
