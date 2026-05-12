/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark theme colors matching GCP console aesthetic
        surface: {
          0: '#0d1117', // page background
          1: '#161b22', // card background
          2: '#1c2128', // elevated surface
          3: '#21262d', // hover state
        },
        border: {
          DEFAULT: '#30363d',
          light: '#484f58',
        },
        text: {
          primary: '#e6edf3',
          secondary: '#8b949e',
          tertiary: '#6e7681',
        },
        accent: {
          blue: '#58a6ff',
          green: '#3fb950',
          red: '#f85149',
          orange: '#d29922',
          purple: '#bc8cff',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Noto Sans',
          'Helvetica',
          'Arial',
          'sans-serif',
        ],
        mono: [
          'SFMono-Regular',
          'Consolas',
          'Liberation Mono',
          'Menlo',
          'Courier',
          'monospace',
        ],
      },
    },
  },
  plugins: [],
};
