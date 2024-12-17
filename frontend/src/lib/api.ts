import axios, { AxiosInstance } from 'axios';

// Types
export interface LoginData {
  email: string;
  password: string;
}

export interface RegisterData extends LoginData {
  phone?: string;
}

export interface EmailPreferences {
  frequency: 'daily' | 'weekly' | 'monthly' | 'workdays' | 'custom';
  time: string;
  customInterval?: number;
}

export interface User {
  id: string;
  email: string;
  phone?: string;
  isAdmin: boolean;
  isActive: boolean;
}

const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const auth = {
  register: (data: RegisterData) =>
    api.post('/api/auth/register', data),
  login: (data: LoginData) =>
    api.post('/api/auth/login', data),
  resetPassword: (email: string) =>
    api.post('/api/auth/reset-password/request', { email }),
  setNewPassword: (token: string, password: string) =>
    api.post('/api/auth/reset-password', { token, password }),
  getCurrentUser: () =>
    api.get('/api/auth/me'),
};

export const user = {
  getDomains: () =>
    api.get('/api/domains'),
  updateDomains: (domains: string[]) =>
    api.post('/api/users/preferences/domains', { domains }),
  getEmailPreferences: () =>
    api.get('/api/users/preferences/email'),
  updateEmailPreferences: (preferences: EmailPreferences) =>
    api.put('/api/users/preferences/email', preferences),
  getReports: (params?: { domain?: string; startDate?: string; endDate?: string }) =>
    api.get('/api/reports', { params }),
  getReportById: (id: string) =>
    api.get(`/api/reports/${id}`),
};

export const admin = {
  getUsers: () =>
    api.get('/api/admin/users'),
  updateUser: (id: string, data: Partial<User>) =>
    api.put(`/api/admin/users/${id}`, data),
  deleteUser: (id: string) =>
    api.delete(`/api/admin/users/${id}`),
  getDomainConfig: () =>
    api.get('/api/admin/domains'),
  updateDomainConfig: (data: { domain: string; dataSources: string[] }) =>
    api.put('/api/admin/domains', data),
  updateLLMConfig: (data: { apiKey: string; baseUrl: string }) =>
    api.put('/api/admin/llm/config', data),
  getLogs: (params?: { level?: string; startDate?: string; endDate?: string }) =>
    api.get('/api/admin/logs', { params }),
  getCrawlerStats: () =>
    api.get('/api/admin/crawler/stats'),
  updateCrawlerFrequency: (frequency: number) =>
    api.put('/api/admin/crawler/frequency', { frequency }),
};

export default api;
