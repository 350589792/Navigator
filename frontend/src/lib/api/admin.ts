import { api } from './base';
import type { User, AdminConfig, LLMConfig, LogResponse, DomainConfig } from '@/types/api';

export const admin = {
  getUsers: async (): Promise<User[]> => {
    const response = await api.get<User[]>('/api/v1/admin/users');
    return response.data;
  },

  updateUser: async (userId: string, data: Partial<User>): Promise<User> => {
    const response = await api.patch<User>(`/api/v1/admin/users/${userId}`, data);
    return response.data;
  },

  updateCrawlerFrequency: async (frequency: string): Promise<AdminConfig> => {
    const response = await api.post<AdminConfig>('/api/v1/admin/crawler/frequency', { frequency });
    return response.data;
  },

  getLogs: async (): Promise<LogResponse> => {
    const response = await api.get<LogResponse>('/api/v1/admin/logs');
    return response.data;
  },

  updateLLMConfig: async (config: LLMConfig): Promise<AdminConfig> => {
    const response = await api.post<AdminConfig>('/api/v1/admin/llm/config', config);
    return response.data;
  },

  updateDomainConfig: async (domain: DomainConfig): Promise<void> => {
    await api.post('/api/v1/admin/domains/config', domain);
  }
};
