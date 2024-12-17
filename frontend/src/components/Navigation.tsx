import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';
import { Link } from 'react-router-dom';

export function Navigation() {
  const { isAuthenticated, isAdmin, logout } = useAuth();
  const { t, language, setLanguage } = useLanguage();

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'zh' : 'en');
  };

  return (
    <nav className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link to="/" className="flex-shrink-0 flex items-center">
              <h1 className="text-xl font-bold">{t('appName')}</h1>
            </Link>
          </div>

          <div className="flex items-center space-x-4">
            <button
              onClick={toggleLanguage}
              className="text-gray-700 hover:text-gray-900 px-3 py-1 rounded-md text-sm font-medium"
            >
              {t('nav.language')}
            </button>

            {isAuthenticated ? (
              <>
                <Link to="/dashboard" className="text-gray-700 hover:text-gray-900">
                  {t('dashboard.title')}
                </Link>
                <Link to="/dashboard/reports" className="text-gray-700 hover:text-gray-900">
                  {t('dashboard.reports.title')}
                </Link>
                {isAdmin && (
                  <div className="flex items-center space-x-4">
                    <Link to="/admin/users" className="text-gray-700 hover:text-gray-900">
                      {t('admin.users.title')}
                    </Link>
                    <Link to="/admin/domains" className="text-gray-700 hover:text-gray-900">
                      {t('admin.domains.title')}
                    </Link>
                    <Link to="/admin/llm" className="text-gray-700 hover:text-gray-900">
                      {t('admin.llm.title')}
                    </Link>
                    <Link to="/admin/logs" className="text-gray-700 hover:text-gray-900">
                      {t('admin.logs.title')}
                    </Link>
                  </div>
                )}
                <button
                  onClick={logout}
                  className="text-gray-700 hover:text-gray-900"
                >
                  {t('nav.logout')}
                </button>
              </>
            ) : (
              <Link to="/auth/login" className="text-gray-700 hover:text-gray-900">
                {t('nav.login')}
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
