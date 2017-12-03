

(defparameter +train-path+ "train.csv")
(defparameter +test-path+ "test.csv")
(defparameter +result-path+ "result.csv")

(defstruct (dataframe (:conc-name df-))
  (label nil)
  (datum #() :type array))

(defun df-index (df label)
  (typecase label
    (number label)
    (string (position label (df-label df) :test #'equal))))

(defun df-column (df label)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (res (make-array num-of-instances))
         (index (df-index df label)))
    (when index
      (loop for i from 0 below num-of-instances do
            (setf (aref res i) (aref (aref datum i) index)))
      res)))

(defun df-select (df labels)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (num-of-features (length labels))
         (res (make-array num-of-instances))
         (indices (make-array num-of-features)))
    (loop for i from 0 below num-of-features do
          (setf (aref indices i) (df-index df (aref labels i))))
    (loop for i from 0 below num-of-instances
          for row = (aref datum i)
          for new-row = (make-array num-of-features) do
          (loop for j from 0 below num-of-features do
                (setf (aref new-row j)
                      (aref row (aref indices j))))
          (setf (aref res i)
                new-row))
    (make-dataframe
      :label labels
      :datum res)))

(defun df-retain (df tester)
  (let ((datum (df-datum df)))
    (make-dataframe
      :label (df-label df)
      :datum (coerce
               (loop for row across datum
                     when (funcall tester row)
                     collect row)
               'vector))))

(defun df-string-to-integer (df label)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (index (df-index df label)))
    (loop for i from 0 below num-of-instances do
          (setf (aref (aref datum i) index)
                (parse-integer (aref (aref datum i) index) :junk-allowed t)))))

(defun df-string-to-float (df label)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (index (df-index df label)))
    (loop for i from 0 below num-of-instances do
          (setf (aref (aref datum i) index)
                (read-from-string (aref (aref datum i) index) nil)))))

(defun df-type-conversion (df types)
  (loop for cast in types
        for index = 0 then (1+ index) do
        (case cast
          ((integer number int)
           (df-string-to-integer df index))
          (float
           (df-string-to-float df index))
          (string nil))))

(defun counts (arr)
  (let ((ht (make-hash-table :test #'equal)))
    (loop for v across arr do
          (setf (gethash v ht)
                (1+ (gethash v ht 0))))
    (let* ((appears (make-hash-table :test #'equal))
           (diff 0)
           (counts (loop for key being each hash-key of ht
                        using (hash-value value)
                        unless (gethash key appears)
                        do (setf (gethash key appears) t
                                 diff (1+ diff))
                        collect (cons key value))))
      (values counts diff))))

(defun df-counts (df)
  (let ((label (df-label df)))
    (if label
        (loop for index across label do
              (multiple-value-bind (counts diff)
                (counts (df-column df index))
                (sort counts #'> :key #'cdr)
                (format t "~A~16T:~16T~A~16T~16T~A ~~~16T~A~%"
                        index diff (first counts) (car (last counts))))))))

(defun df-predict (df predict)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (result (make-array num-of-instances)))
    (loop for i from 0 below num-of-instances do
          (setf (aref result i)
                (funcall predict (aref datum i))))
    result))

(defun split (str delim)
  (let ((res (make-array 0 :element-type 'string
                           :fill-pointer 0
                           :adjustable t))
        (dlen (length delim)))
    (loop for i from 0 below (length str)
          for ch = (char str i)
          with dc = 0
          with word-start = 0
          with delim-start = -1
          with lq = nil
          with q = nil
          do
          (if (null q)
              (if (or (eq ch #\")
                      (eq ch #\'))
                  (setf q t
                        lq ch)
                  (if (eq ch (char delim dc))
                      (if (<= 0 delim-start)
                          (progn
                            (incf dc)
                            (when (= dc dlen)
                              (vector-push-extend (subseq str word-start delim-start) res)
                              (setf dc 0
                                    word-start (1+ i)
                                    delim-start -1)))
                          (progn
                            (setf delim-start i
                                  dc (1+ dc))
                            (when (= dc dlen)
                              (vector-push-extend (subseq str word-start delim-start) res)
                              (setf dc 0
                                    word-start (1+ i)
                                    delim-start -1))))
                      (when (< 0 dc)
                        (setf delim-start -1
                              dc 0))))
              (if (eq ch lq)
                  (setf q nil)))
          finally
          (if (< word-start i)
              (vector-push-extend (subseq str word-start i) res)))
    res))

(defun loadtxt (path delimiter)
  (let ((datum (make-array 0 :element-type 'array
                             :fill-pointer 0
                             :adjustable t)))
    (with-open-file (in path :direction :input)
      (loop for line = (read-line in nil nil) while line do
            (vector-push-extend (split line delimiter) datum)))
    datum))

(defun loadcsv (path &key header)
  (let ((data (loadtxt path ",")))
    (if header
        (make-dataframe :label (aref data 0)
                        :datum (subseq data 1))
        (make-dataframe :datum data))))

(defun make-naive-bayes-classifier (classes feature-labels feature-counts feature-likelihood class-counts class-likelihood)
  (lambda (data)
    (let ((result (make-hash-table :test #'equal)))
      (dolist (class classes)
        (loop for label across feature-labels
              for value across data do

;              (format t "P(~A|~A=~A) = ~A/~A * ~A / ~A~%"
;                      class label value
;                      (gethash (cons class value) feature-counts 0)
;                      (gethash class class-counts)
;                      (gethash class class-likelihood)
;                      (gethash (cons label value) feature-likelihood))

              (setf (gethash class result)
                    (* (gethash class result 1)
                       (/ (* (/ (gethash (cons class value) feature-counts 0)
                                (gethash class class-counts))
                             (gethash class class-likelihood))
                          (gethash (cons label value) feature-likelihood 1))))))
      (let ((infer nil)
            (probabilities (list)))
        (loop for key being each hash-key of result
              using (hash-value value) do
              (push (cons key value) probabilities))
        (setf infer
              (car (first (sort (copy-list probabilities) #'> :key #'cdr))))
        (values infer probabilities)))))

(defun naive-bayes-classifier (df class-index feature-labels)
  (let* ((datum (df-datum df))
         (class-index (df-index df class-index))
         (num-of-instances (length datum))
         (classes (list))
         (feature-counts (make-hash-table :test #'equal))
         (class-counts (make-hash-table :test #'equal))
         (class-likelihood (make-hash-table :test #'equal))
         (feature-likelihood (make-hash-table :test #'equal)))
    (dolist (pair (counts (df-column df class-index)))
      (let* ((class-value (car pair))
             (class-count (cdr pair))
             (class-contents (df-retain df (lambda (row) (equal class-value (aref row class-index))))))
        (push class-value classes)
        (setf (gethash class-value class-counts)
              class-count)
        (setf (gethash class-value class-likelihood)
              (/ class-count num-of-instances))
        (loop for label across feature-labels do
          (dolist (fpair (counts (df-column class-contents label)))
            (let* ((feature-value (car fpair))
                   (feature-count (cdr fpair))
                   (c-f (cons class-value feature-value))
                   (l-f (cons label feature-value))
                   (ratio (/ feature-count num-of-instances)))
              (setf (gethash c-f feature-counts)
                    feature-count
                    (gethash l-f feature-likelihood)
                    (+ (gethash l-f feature-likelihood 0) ratio)))))))
    (make-naive-bayes-classifier classes feature-labels feature-counts feature-likelihood class-counts class-likelihood)))

(defun print-array (stream array &optional (delim " "))
  (let ((num-of-elements (length array)))
    (princ (aref array 0) stream)
    (loop for i from 1 below num-of-elements do
          (format stream "~A~A" delim (aref array i)))))

(let ((df (loadcsv +train-path+ :header t))
      (test-df (loadcsv +test-path+ :header t)))
  (df-type-conversion df '(int int int string string float int int string float string string))
  (df-type-conversion test-df '(int int string string float int int string float string string))
  (terpri)
  (df-counts df)
  (let* ((datum (df-datum df))
         (test-datum (df-datum test-df))
         (train-survived (df-index df "Survived"))
         (test-passenger-id (df-index test-df "PassengerId"))
         (using-labels #("Sex" "Age" "SibSp" "Parch" "Cabin"))
         (classify (naive-bayes-classifier df "Survived" using-labels))
         (train-df (df-select df using-labels))
         (train-results (df-predict train-df classify))
         (using-test-df (df-select test-df using-labels))
         (test-results (df-predict using-test-df classify))
         (correct 0))
    (loop for i from 0 below (length train-results) do
          (if (equal (aref (aref datum i) train-survived)
                     (aref train-results i))
          (incf correct)))
    (format t "Train correct ~A/~A(~4F)~%"
            correct (length datum) (/ correct (length datum)))
    (with-open-file (out +result-path+ :direction :output :if-exists :supersede)
      (print-array out #("PassengerId" "Survived") ",")
      (terpri out)
      (loop for i from 0 below (length test-results) do
            (format out "~A,~A~%"
                    (aref (aref test-datum i) test-passenger-id)
                    (aref test-results i))))
    ))


