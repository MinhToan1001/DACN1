import mongoose from 'mongoose';

const projectSchema = new mongoose.Schema({
  projectName: {
    type: String,
    required: true
  },
  totalDonations: {
    type: Number,
    default: 0
  },
  targetDonations: {
    type: Number,
    default: 1000
  }
});

const Project = mongoose.model('Project', projectSchema);
export { Project };
