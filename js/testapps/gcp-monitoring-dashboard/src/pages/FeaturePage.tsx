import { ArrowLeft } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';

export function FeaturePage() {
  const { featureName } = useParams<{ featureName: string }>();

  return (
    <div>
      {/* Breadcrumb */}
      <div className="mb-6">
        <div className="mb-1 flex items-center gap-2 text-sm text-text-secondary">
          <Link to="/" className="hover:text-accent-blue">
            Genkit
          </Link>
          <span>›</span>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/" className="text-text-secondary hover:text-text-primary">
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <h2 className="text-xl font-semibold">{featureName}</h2>
        </div>
      </div>

      {/* Placeholder content */}
      <div className="card flex items-center justify-center p-12">
        <p className="text-text-tertiary">
          Feature detail page — will show metrics, charts, and traces for{' '}
          <span className="text-text-secondary">{featureName}</span>
        </p>
      </div>
    </div>
  );
}
