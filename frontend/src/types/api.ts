export interface AuthResponse {
  token: string;
  user: User;
}

export interface User {
  id: string;
  email: string;
  name: string;
  isAdmin: boolean;
  status?: 'active' | 'disabled';
}

export interface LoginData {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
  phone?: string;
}

export interface ResetPasswordData {
  email: string;
}

export interface UpdatePasswordData {
  token: string;
  password: string;
}

export interface DomainPreference {
  userId: string;
  domains: string[];
}

export interface Report {
  id: string;
  date?: string;
  domains?: string[];
  content?: string;
  summary?: string;
  createdAt?: string;
  status?: 'processing' | 'completed' | 'error';
  error?: string;
}

export interface AdminConfig {
  crawlerFrequency: string;
  llmConfig: LLMConfig;
  domains: DomainConfig[];
}

export interface LLMConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
}

export interface Log {
  id: string;
  timestamp: string;
  level: string;
  message: string;
  metadata: Record<string, any>;
}

export interface LogResponse {
  data: Log[];
  total: number;
}

export interface DomainConfig {
  domain: string;
  dataSources: string[];
}

export interface EmailPreference {
  userId: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  time: string;
  enabled: boolean;
}
