import axios from 'axios';
import { NewsItem } from '../types/NewsItem';

export const fetchNews = async (): Promise<NewsItem[]> => {
  const res = await axios.get('http://localhost:5001/api/news');
  return res.data;
};
