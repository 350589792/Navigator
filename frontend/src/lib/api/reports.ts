import { api } from './base';
import type { Report } from '@/types/api';

export const reports = {
  getReports: async (): Promise<Report[]> => {
    const response = await api.get<Report[]>('/reports');
    return response.data;
  },

  getReportById: async (id: string): Promise<Report> => {
    const response = await api.get<Report>(`/reports/${id}`);
    return response.data;
  },

  generateReport: async (): Promise<Report> => {
    const response = await api.post<Report>('/reports/generate');
    return response.data;
  }
};
