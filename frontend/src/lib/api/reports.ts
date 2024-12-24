import { api } from './base';
import type { Report } from '@/types/api';
import { AxiosError } from 'axios';
import { auth } from './auth';

export interface GenerateReportOptions {
  format: 'text' | 'html';
  domains: string[];
  preferences: {
    keywords: boolean;
    trends: boolean;
    keyPoints: boolean;
  };
}

export const reports = {
  getReports: async (): Promise<Report[]> => {
    try {
      const response = await api.get<Report[]>('/reports');
      return response.data;
    } catch (error) {
      if (error instanceof AxiosError && error.response?.status === 401) {
        try {
          // Try to refresh the token
          await auth.refreshToken();
          // Retry the request
          const retryResponse = await api.get<Report[]>('/reports');
          return retryResponse.data;
        } catch (refreshError) {
          console.error('Error refreshing token:', refreshError);
          throw error;
        }
      }
      console.error('Error fetching reports:', error);
      return [];
    }
  },

  getReportById: async (id: string): Promise<Report | null> => {
    try {
      const response = await api.get<Report>(`/reports/${id}`);
      return response.data;
    } catch (error) {
      if (error instanceof AxiosError && error.response?.status === 401) {
        try {
          await auth.refreshToken();
          const retryResponse = await api.get<Report>(`/reports/${id}`);
          return retryResponse.data;
        } catch (refreshError) {
          console.error('Error refreshing token:', refreshError);
          return null;
        }
      }
      console.error(`Error fetching report ${id}:`, error);
      return null;
    }
  },

  generateReport: async (options: GenerateReportOptions): Promise<Report | null> => {
    try {
      const response = await api.post<Report>('/reports/generate', options);
      return response.data;
    } catch (error) {
      if (error instanceof AxiosError && error.response?.status === 401) {
        try {
          await auth.refreshToken();
          const retryResponse = await api.post<Report>('/reports/generate', options);
          return retryResponse.data;
        } catch (refreshError) {
          console.error('Error refreshing token:', refreshError);
          return null;
        }
      }
      console.error('Error generating report:', error);
      return null;
    }
  }
};
