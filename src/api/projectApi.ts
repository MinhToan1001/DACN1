import axios from 'axios';
import { Project } from '../types/Project';

export const fetchProject = async (): Promise<Project[]> => {
  const res = await axios.get('http://localhost:5001/api/projects');
  return res.data;
};
