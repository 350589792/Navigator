import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '@/contexts/LanguageContext';
import realtimeIcon from '@/assets/images/feature-icons/realtime.svg';
import aiIcon from '@/assets/images/feature-icons/ai.svg';
import reportsIcon from '@/assets/images/feature-icons/reports.svg';
import analyticsIcon from '@/assets/images/feature-icons/analytics.svg';
import heroBg from '@/assets/images/hero-bg.svg';

const features = [
  {
    key: 'realtime',
    icon: realtimeIcon,
  },
  {
    key: 'ai',
    icon: aiIcon,
  },
  {
    key: 'reports',
    icon: reportsIcon,
  },
  {
    key: 'analytics',
    icon: analyticsIcon,
  }
];

export default function IntroductionPage() {
  const navigate = useNavigate();
  const { t, language, setLanguage } = useLanguage();

  return (
    <div
      className="min-h-screen bg-no-repeat bg-cover bg-center text-white"
      style={{ backgroundImage: `url(${heroBg})` }}
    >
      <nav className="container mx-auto px-4 py-6 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-primary">
          {t('appName')}
        </h1>
        <div className="flex gap-4">
          <Button
            variant="ghost"
            onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
          >
            {t('nav.language')}
          </Button>
          <Button onClick={() => navigate('/auth/login')}>
            {t('nav.login')}
          </Button>
        </div>
      </nav>

      <div className="container mx-auto px-4 py-20">
        <div className="text-center">
          <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
            {t('hero.title')}
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            {t('hero.subtitle')}
          </p>
          <Button
            size="lg"
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
            onClick={() => navigate('/auth/register')}
          >
            {t('hero.getStarted')} <ArrowRight className="ml-2" />
          </Button>
        </div>
      </div>

      <div className="container mx-auto px-4 py-16">
        <h2 className="text-3xl font-bold text-center mb-12 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
          {t('features.title')}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature) => (
            <Card key={feature.key} className="bg-gray-800/50 backdrop-blur-lg border-gray-700 hover:bg-gray-800/70 transition-all duration-300">
              <CardHeader>
                <div className="flex items-center justify-center mb-4">
                  <img src={feature.icon} alt={t(`features.${feature.key}.title`)} className="w-16 h-16" />
                </div>
                <CardTitle className="text-center text-white">
                  {t(`features.${feature.key}.title`)}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-center text-gray-300">
                  {t(`features.${feature.key}.description`)}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      <div className="container mx-auto px-4 py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center transform hover:scale-105 transition-transform duration-300">
            <div className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600 mb-2">100+</div>
            <div className="text-gray-300">{t('stats.dataSources')}</div>
          </div>
          <div className="text-center transform hover:scale-105 transition-transform duration-300">
            <div className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600 mb-2">24/7</div>
            <div className="text-gray-300">{t('stats.realtime')}</div>
          </div>
          <div className="text-center transform hover:scale-105 transition-transform duration-300">
            <div className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600 mb-2">10+</div>
            <div className="text-gray-300">{t('stats.domains')}</div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-16 text-center">
        <h2 className="text-3xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
          {t('cta.title')}
        </h2>
        <Button
          size="lg"
          className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
          onClick={() => navigate('/auth/register')}
        >
          {t('cta.action')} <ArrowRight className="ml-2" />
        </Button>
      </div>
    </div>
  );
}
